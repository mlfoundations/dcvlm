import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from vlmeval.utils.dmt_utils import get_ckpt_code

FAIL_MSG = 'Failed to obtain answer via API.'

# settings the max_new_tokens hp according to the benchmark
DEFAULT_MNT = 128
MNT_PER_DATASET = dict(
    # we are good with 128 by default, these are exceptions so far
    COCO=16,
    OlympiadBench=4096,
    WeMath=4096,
    LogicVista=4096,
    MMVet=4096,
    DynaMath=4096,
    MathVista=4096,
    MathVision=4096,
    MathVerse=4096,
    WildVision=4096,
    CharXiv=4096,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, load_dmt_pretrained='', generation_config={}):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    # keeping these as kwargs so we add stuff easily later on
    if generation_config['max_new_tokens'] == -1:
        found = False
        for _dataset, max_new_tokens in MNT_PER_DATASET.items():
            if listinstr([_dataset], dataset_name):
                found = True
                break
        
        if not found:
            max_new_tokens = DEFAULT_MNT
        
        generation_config['max_new_tokens'] = max_new_tokens
        if rank == 0:
            print(
                "max_new_tokens is not set from cmd-line",
                f"setting max_new_tokens={max_new_tokens} for {dataset_name}.",
                sep=" "
            )
    
    init_kwargs = {'load_dmt_pretrained': load_dmt_pretrained, **generation_config}
    model = supported_VLM[model_name](**init_kwargs) if isinstance(model, str) else model

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), disable=False):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, load_dmt_pretrained='', overwrite=False, generation_config={}):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    if load_dmt_pretrained:
        # this is how local files are modified to read from ckpts
        ckpt_code = get_ckpt_code(pretrained_path=load_dmt_pretrained)
        result_file = osp.join(work_dir, f"{model_name}_{ckpt_code}_{dataset_name}.xlsx")
        prev_file = f'{work_dir}/{model_name}_{ckpt_code}_{dataset_name}_PREV.pkl'
        tmpl = osp.join(work_dir, '{}' + f'{world_size}_{model_name}_{ckpt_code}_{dataset_name}.pkl')
    else:
        # this is how local files were originally saved in VLMEvalKit
        result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')
        prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
        tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    
    # this will dump already exising input-output lines to :prev_file:, so the :infer_data:
    # function reads from it and avoids running inference
    if osp.exists(result_file) and not overwrite:
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, 
        work_dir=work_dir, 
        model_name=model_name, 
        dataset=dataset,
        out_file=out_file, 
        verbose=verbose, 
        api_nproc=api_nproc, 
        load_dmt_pretrained=load_dmt_pretrained,
        generation_config=generation_config
    )
    
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    
    if world_size > 1:
        dist.barrier()
    return model
