import argparse
import itertools
import json
import os
import random
from functools import partial
from PIL import Image
import pillow_avif
import pandas as pd

import torch
import torch.distributed as dist
from transformers import set_seed

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from tqdm import tqdm

from dmt_evals.vlmeval.utils.dmt_utils import get_ckpt_code

# all possible splits for MLLM-Guard
ALL_SPLITS = [
    "privacy",
    "bias",
    "toxicity",
    "noise-injection",
    "position-swapping",
    "hallucination",
    "legality"
]

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    data_items = [_['data_item'] for _ in batches]
    return pixel_values, data_items


def combine_question_with_options(x):
    lan2please_answer = {'zh': "(请用选项回答)", 'en': "(Please answer with options)"}
    # question = x['question'] + lan2please_answer[x['Lan']] + " " + x['Options'].strip()
    # always English instruction in original eval
    question = x['question'] + lan2please_answer['en'] + " " + x['Options']
    return question


def get_noised_image_file(file_name):
    file_name, ext = os.path.splitext(file_name)
    return f'{file_name}_noise{ext}'


class MLLMGuardDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):

        self.images_dir = os.path.join(root, split, 'imgs')
        self.ann_path = os.path.join(root, split, 'prompt.csv')

        # Create a df where the rows are interleaved (file_name then Reverse Image)
        df = pd.read_csv(self.ann_path)
        df['category'] = df.apply(lambda x: f"{x['Category I']}/{x['Jailbreak Type']}", axis=1)
        df['split'] = split

        if split == 'position-swapping':
            df = df[['file_name', 'Reverse Image', 'Prompt', 'category', 'split']]
        elif split == 'hallucination':
            df = df[['file_name', 'Prompt', 'Options', 'category', 'split', 'Lan']]
        else:
            df = df[['file_name', 'Prompt', 'category', 'split']]
        df = df.rename(columns={'Prompt': 'question'})

        if split == 'position-swapping':
            df1 = df[['file_name', 'question', 'category', 'split']]
            df2 = df[['Reverse Image', 'question', 'category', 'split']]
            df2 = df2.rename(columns={'Reverse Image': 'file_name'})

            df1.index = range(0, len(df1) * 2, 2)
            df2.index = range(1, len(df2) * 2, 2)

            df = pd.concat([df1, df2]).sort_index().reset_index(drop=True)
        elif split == 'hallucination':
            df1 = df[['file_name', 'question', 'category', 'split']]
            df2 = df[['file_name', 'question', 'Options', 'category', 'split', 'Lan']]

            df2['question'] = df2.apply(lambda x: combine_question_with_options(x), axis=1)
            df2 = df2.drop(columns=['Options', 'Lan'])

            df1.index = range(0, len(df1) * 2, 2)
            df2.index = range(1, len(df2) * 2, 2)

            df = pd.concat([df1, df2]).sort_index().reset_index(drop=True)
        elif split == 'noise-injection':
            df1 = df.copy()
            df2 = df.copy()
            df2['file_name'] = df2.apply(lambda x: get_noised_image_file(x['file_name']), axis=1)

            df1.index = range(0, len(df1) * 2, 2)
            df2.index = range(1, len(df2) * 2, 2)

            df = pd.concat([df1, df2]).sort_index().reset_index(drop=True)

        # Remove image dir prefix in file name because added later (note: original "file_name" has the dir but "Reverse Image" does not)
        df['file_name'] = df.apply(lambda x: x['file_name'].replace('imgs/', ''), axis=1)

        self.data = df.to_dict(orient='records')

        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        image_path = os.path.join(self.images_dir, data_item['file_name'])
        data_item['index'] = idx

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f"{image_path} can't be opened!!")
            exit(1)

        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'pixel_values': pixel_values,
            'data_item': data_item
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    # reproducible setup    
    set_seed(args.seed)

    # save results in the corresponding seed subfolder 
    if not args.results_dir.endswith(f"seed_{args.seed}"):
        args.results_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
    os.makedirs(args.results_dir, exist_ok=True)

    # start looping through all dataset splits
    splits = ALL_SPLITS if args.datasets == "all" else args.datasets.split(",")
    for split in splits:
        dataset = MLLMGuardDataset(
            root=args.root,
            split=split,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        rank, world_size = dist.get_rank(), dist.get_world_size()
        for _, (pixel_values, data_items) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{split}, rank {rank+1}/{world_size}"):
            question = data_items[0]['question']

            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                do_sample=False, # always use greedy decoding
                temperature=0.0,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                verbose=True
            )

            data_item = data_items[0]
            data_item['prediction'] = pred
            outputs.append(data_item)

        # wait for all processes to finish inference and gather results
        dist.barrier()
        merged_outputs = [None for _ in range(world_size)]
        dist.all_gather_object(merged_outputs, json.dumps(outputs))
        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if rank == 0:
            # Convert results into an .xlsx file
            df = pd.DataFrame(merged_outputs)
            df = df.set_index('index')
            df = df.sort_index()

            # create the results_dir with the model name
            results_dir = os.path.join(args.results_dir, args.model, "MLLMGuard")
            os.makedirs(results_dir, exist_ok=True)

            # tag the results file with the ckpt name (consistent with vlmevalkit dumping)
            ckpt_code = get_ckpt_code(args.checkpoint)
            results_file = os.path.join(results_dir, f'{args.model}_{ckpt_code}_MLLMGuard_{split}{args.results_ext}')
            df.to_excel(results_file, index=False) if args.results_ext == '.xlsx' else df.to_csv(results_file, index=False)
            print('Results saved to {}'.format(results_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument('--model', type=str, required=True, 
                        help="Model string, so result files are saved in the appropriate subfolder.")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    
    # dataloading args
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    
    # inference args (some are unneeded cuz we always use greedy decoding)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--dynamic', action='store_true', help="Whether or not to use dynamic resolution.")
    parser.add_argument('--max-num', type=int, default=6, help="Maximum number of image tiles (excluding thumbnail).")
    parser.add_argument('--seed', type=int, default=1234)

    # eval args 
    parser.add_argument('--root', type=str, default='data/eval/mllmguard/desensitize', 
                        help='Path to where MLLM-Guard is saved on disk.')
    parser.add_argument('--datasets', type=str, default='all', 
                        help='Comma separated list of dataset splits for inference. Pass \'all\' to run on all splits.')
    parser.add_argument('--results_dir', type=str, default='eval/mllmguard/eval_results')
    parser.add_argument('--results_ext', type=str, default='.csv')

    args = parser.parse_args()
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    # initialize distributed mode (must launch this script with torchrun)
    env_local_rank = int(os.getenv('LOCAL_RANK', '0')) 
    env_rank = int(os.getenv('RANK', '0')) 
    env_world_size = int(os.getenv('WORLD_SIZE', '1'))
    dist.init_process_group(
        backend='nccl',
        world_size=env_world_size,
        rank=env_rank,
        device_id=torch.device(f"cuda:{env_local_rank}")
    )
    torch.cuda.set_device(env_local_rank)

    model, tokenizer = load_model_and_tokenizer(
        checkpoint=args.checkpoint,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        auto=args.auto
    )
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    args.num_beams = 1
    print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()

    if dist.is_initialized():
        dist.destroy_process_group()
