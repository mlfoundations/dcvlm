"""
Simple sweep on all our supported benchmarks to download them on disk.
Also checks which dataset type and eval method they support, may not be accurate because
some datasets implement the `evaluate` internally. 
"""
from vlmeval.dataset import build_dataset, DMT_SUPPORTED_DATASETS
from vlmeval.smp import listinstr


def download_and_log():

    for category, datasets in DMT_SUPPORTED_DATASETS.items():
        for dataset_name in datasets:
            
            judge_kwargs = {
                'nproc': 4,
                'verbose': True,
                'retry': 3
            }
            
            dataset = build_dataset(dataset_name)
            if dataset is None:
                print(f"{dataset_name} is not supported in vlmevalkit.")
                continue
            elif dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro']:
                judge_kwargs['model'] = 'chatgpt-0125'
            elif listinstr(['MMVet', 'LLaVABench', 'MMBench-Video'], dataset_name):
                judge_kwargs['model'] = 'gpt-4-turbo'
            elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'WeMath', 'LogicVista'], dataset_name):  # noqa: E501
                judge_kwargs['model'] = 'gpt-4o-mini'
            elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision'], dataset_name):  # noqa: E501
                judge_kwargs['model'] = 'gpt-4o'

            print(f"{dataset_name} - {dataset.TYPE} - {judge_kwargs.get('model', 'exact_matching')}")



if __name__ == "__main__":
    download_and_log()