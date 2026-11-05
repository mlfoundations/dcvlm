import argparse
import itertools
import json
import os
import time
from functools import partial
from PIL import Image

import torch
import torch.distributed as dist
from transformers import set_seed
from tqdm import tqdm

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from dmt_evals.vlmeval.utils.dmt_utils import get_ckpt_code



def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    data_items = [_['data_item'] for _ in batches]
    return pixel_values, data_items


def load_data(subset, metaDir):
    with open(metaDir, 'r') as f:
        all_questions = json.load(f)
    if subset == 'unsafes':
        all_questions = [q for q in all_questions if q['safe'] == False]
        for question in all_questions:
            question['question'] = question['instr-resp'][0]['instruction']
    else:
        all_questions = [q for q in all_questions if q['safe'] == True]
        key = 'unsafe_instruction' if subset == 'safe_unsafes' else 'safe_instruction'
        for question in all_questions:
            question['question'] = next(
                (q[key] for q in question['instr-resp'] if key in q), None
            )
    return all_questions


class VLGuardDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 root, 
                 metadata_dir, 
                 split, 
                 input_size=224, 
                 dynamic_image_size=False,
                 use_thumbnail=False, 
                 max_num=6):
        
        self.data = load_data(split, metadata_dir)
        self.root_path = root

        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        img_id = data_item['image']
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')

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
            'data_item': data_item,
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

    score_files = []
    for ds_name in args.datasets:
        dataset = VLGuardDataset(
            root=args.imageDir,
            metadata_dir=args.metaDir,
            split=ds_name,
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
        for _, (pixel_values, data_items) in tqdm(enumerate(dataloader), total=len(dataloader)):
            question = data_items[0]['question']

            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                verbose=True
            )

            data_item = data_items[0]
            data_item['response'] = pred
            outputs.append(data_item)

        dist.barrier()

        world_size = dist.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        dist.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if dist.get_rank() == 0:
            # collect inp-out pairs
            temp = {}
            for data_item in merged_outputs:
                image_id = data_item['image']
                temp[image_id] = data_item

            # folder consistent with vlmevalkit
            out_dir = os.path.join(args.out_dir, args.model)
            os.makedirs(out_dir, exist_ok=True)
            
            # filename consistent with vlmevalkit
            ckpt_code = get_ckpt_code(args.checkpoint)
            results_base = f'{args.model}_{ckpt_code}_VLGuard-{ds_name}.json'
            output_path = os.path.join(out_dir, results_base)
            
            with open(output_path, 'w') as file_ptr:
                json.dump(temp, file_ptr, indent=4)
                print(f'Results saved to {output_path}')

            # we can compute ASR in the same script since it
            # doesn't call API models, otherwise we just leave 
            # helpfulness as a separate script
            # (e.g. to call on the lrd_all_serial partition)
            score_file = results_base.replace('.json', '_score.json')
            if ds_name != 'safe_safes':
                cmd = (
                    "python eval/vlguard/calculate_asr.py  "
                    f"--split {ds_name} "
                    f"--output_file '{results_base}' "
                    f"--output_dir {out_dir} "
                    f"--score_file '{score_file}' "
                )
                print(f"About to run the following script:\n{cmd}")
                os.system(cmd)
            
            score_files.append(os.path.join(out_dir, score_file))
    
    if dist.get_rank() == 0:
        overall_scores = {}
        for dataset, score_file in zip(args.datasets, score_files):
            with open(score_file, "r") as f:
                score = json.load(f)
            if "ASR" in score:
                overall_scores[f"{dataset}_ASR"] = score["ASR"]
        
        overall_score_file = os.path.join(out_dir, f"{args.model}_{ckpt_code}_VLGuard_overall.json")
        with open(overall_score_file, "w") as f:
            json.dump(overall_scores, f)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    
    # data / dataloader args
    parser.add_argument('--datasets', type=str, default='safe_safes,safe_unsafes,unsafes')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    
    # generation args
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-new-tokens', default=128, type=int, help='Max tokens for generation.')

    # eval folders
    parser.add_argument('--out-dir', type=str, default='eval/vlguard/results')
    parser.add_argument('--metaDir', default='data/eval/vlguard/test.json', type=str)
    parser.add_argument('--imageDir', default='data/eval/vlguard/test', type=str)

    args = parser.parse_args()
    set_seed(args.seed)

    # make sure all outputs are within the seed folder (e.g. outputs/seed_1234/<blabla>.csv)
    base_outdir = os.path.basename(args.out_dir).removesuffix("/")
    seed_dir = f"seed_{args.seed}"
    if base_outdir != seed_dir:
        args.out_dir = os.path.join(args.out_dir, seed_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    
    # checks for args
    assert args.batch_size == 1, 'Only batch size 1 is supported'
    assert args.dynamic, "We should always use dynamic resolution!"
    assert args.num_beams == 1 and args.temperature == 0, "Please use greedy decoding!"

    # setup distributed environment
    # we always work on a single node here
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    assert rank == local_rank, "Are you trying multi-node?"
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}")
    )
    torch.cuda.set_device(local_rank)

    model, tokenizer = load_model_and_tokenizer(
        checkpoint=args.checkpoint,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        auto=args.auto
    )
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()

    # safe exit :)
    if dist.is_initialized():
        dist.destroy_process_group()