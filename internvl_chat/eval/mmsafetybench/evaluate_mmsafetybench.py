import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from PIL import Image

import torch
import torch.distributed as dist
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from tqdm import tqdm

from transformers import set_seed
from dmt_evals.vlmeval.utils.dmt_utils import get_ckpt_code

splits_list = [
    "SD",
    "SD_TYPO",
    "TYPO"
]

scenario_list = [
    "01-Illegal_Activitiy",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision"
]

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    data_items = [_['data_item'] for _ in batches]
    return pixel_values.to(torch.bfloat16), data_items


class MMSafetyBenchDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 questions_root, 
                 images_root, 
                 scenario, 
                 split, 
                 input_size=224, 
                 dynamic_image_size=False,
                 use_thumbnail=False, 
                 max_num=6):

        self.images_dir = os.path.join(images_root, scenario, split)

        questions_path = os.path.join(questions_root, scenario + ".json")
        with open(questions_path, 'r') as f:
            questions = json.load(f)
        self.questions = questions
        self.question_ids = list(questions.keys())

        self.question_key = 'Rephrased Question' if 'TYPO' in split else 'Rephrased Question(SD)'

        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        idx = self.question_ids[idx]
        data_item = self.questions[idx]
        image_path = os.path.join(self.images_dir, f'{idx}.jpg')
        data_item['pid'] = idx
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
    # reproducibility 
    set_seed(args.seed)

    # create results folder with seed
    if not args.results_dir.endswith(f"seed_{args.seed}"):
        args.results_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
    os.makedirs(args.results_dir, exist_ok=True)

    # lists of splits (e.g. SD only, TYPO or SD+TYPO) and scenarios for inference
    splits_for_inference = splits_list if args.splits == "all" else args.splits.split(',')
    scenarios_for_inference = scenario_list if args.scenarios == "all" else args.scenarios.split(',')

    # inject this variable (useful later on)
    args.ckpt_code = get_ckpt_code(args.checkpoint)

    # define params for decoding
    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=1,
        do_sample=False, # always use greedy decoding
        temperature=0.0,
    )

    # start running inference on splits x scenarios
    for split in splits_for_inference:
        for scenario in scenarios_for_inference:
            
            dataset = MMSafetyBenchDataset(
                questions_root=args.questions_dir,
                images_root=args.images_dir,
                scenario=scenario,
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
                prefetch_factor=args.num_workers
            )

            # naming and dumping consistent with vlmevalkit
            # (skip if computed already)
            out_dir = os.path.join(args.results_dir, args.model, "MMSafetyBench", scenario)
            os.makedirs(out_dir, exist_ok=True)
            results_file = f'{args.model}_{args.ckpt_code}_MMSafetyBench_{split}.json'
            output_path = os.path.join(out_dir, results_file)
            if os.path.isfile(output_path): 
                with open(output_path, 'r') as in_ptr:
                    data = json.load(in_ptr)
                if len(data) == len(dataset):
                    print(f"Skipping {split} x {scenario} because it exists at {output_path}.")
                    continue

            outputs = []
            iterator = tqdm(
                enumerate(dataloader), 
                total=len(dataloader), 
                desc=f"{split}-{scenario}, rank {dist.get_rank()+1}/{dist.get_world_size()}",
            )
            for _, (pixel_values, data_items) in iterator:
                question = data_items[0][dataset.question_key]
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values.cuda(),
                    question=question,
                    generation_config=generation_config,
                    verbose=False
                )

                data_item = data_items[0]
                data_item['response'] = pred
                outputs.append(data_item)

            # wair for all ranks to finish and gather results
            dist.barrier()
            world_size = dist.get_world_size()
            merged_outputs = [None for _ in range(world_size)]
            dist.all_gather_object(merged_outputs, json.dumps(outputs))
            merged_outputs = [json.loads(_) for _ in merged_outputs]
            merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

            # dump inp/out pairs on disk
            # will use GPT-4 for evaluation in calculate_score.py
            if dist.get_rank() == 0:
                temp = {}
                for data_item in merged_outputs:
                    pid = data_item['pid']
                    temp[pid] = data_item
                
                # dumping
                with open(output_path, 'w') as out_ptr:
                    json.dump(temp, out_ptr, indent=4)
                print(f'Results saved to {output_path}')

                # NOTE: commenting this out, gotta run on different partition cuz APIs
                # cmd = f'python eval/mmsafetybench/calculate_score.py --scenario {scenario} --results_path {output_path} --out_dir {args.out_dir} --max_num_retries {args.max_num_retries} --llm_engine {args.llm_engine} --max_new_gpt_tokens {args.max_new_gpt_tokens}'
                # print(cmd)
                # os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    
    # dataloader & dataset args
    parser.add_argument('--splits', type=str, default='all')
    parser.add_argument('--scenarios', type=str, default='all')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--questions-dir', type=str, 
                        default='data/eval/mmsafetybench/processed_questions', 
                        help='path to questions dir')
    parser.add_argument('--images-dir', type=str, 
                        default='data/eval/mmsafetybench/imgs', 
                        help='path to images dir')
    parser.add_argument('--results-dir', type=str, 
                        default='eval/mmsafetybench/questions_with_answers', 
                        help='path to generated outputs')

    # decoding & inference args
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    
    # misc
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    # setup distributed mode 
    env_local_rank = int(os.getenv('LOCAL_RANK', 0))
    env_rank = int(os.getenv('RANK', 0))
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
