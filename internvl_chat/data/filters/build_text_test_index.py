import random
import os
import time
import numpy as np
import json
from glob import glob
import re
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from argparse import ArgumentParser
import pickle
from datasketch import MinHash, MinHashLSH  # note this lib has also been used for deduplicating pile: https://github.com/EleutherAI/pile_dedupe
from collections import OrderedDict
import torch

# Global LSH object for worker processes
global_lsh = None

# These datasets can be downloaded using `wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip & unzip OpenCompassData-complete-20240207.zip`
DATA_ROOT = os.environ.get("TEXT_EVAL_DATA_ROOT", "/path/to/OpenCompassData/data")

# preprocessed text-only training sets (see data/preprocessing), under $DATA_ROOT
_TRAIN_DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/datacomp_data")
TRAIN_ROOTS = {
    "flan": "flan/preprocessed_9.jsonl",
    "flanv2": "flanv2/preprocessed_9.jsonl",
    "mathqa": "mathqa/preprocessed.jsonl",
    "gsm8k-socratic": "gsm8k-socratic/preprocessed.jsonl",
    "dolly": "dolly/preprocessed.jsonl",
    "ultrafeedback": "ultrafeedback/preprocessed.jsonl",
    "xcoder_80k": "xcoder_80k/preprocessed.jsonl",
    "ultrachat_200k": "ultrachat_200k/preprocessed.jsonl",
    "slimorca": "slimorca/preprocessed.jsonl",
    "numina-math-1.5": "numina_math_1.5/preprocessed.jsonl",
}
TRAIN_ROOTS = {k: os.path.join(_TRAIN_DATA_ROOT, v) for k, v in TRAIN_ROOTS.items()}

PARAM_SWEEPS = [

    (5, 0.1, 128),
    (5, 0.2, 128),
    (5, 0.3, 128),
    (5, 0.4, 128),
    (5, 0.5, 128),
    (5, 0.6, 128),
    (5, 0.7, 128),
    (5, 0.8, 128),
    (5, 0.9, 128),

    (8, 0.4, 128),
    (8, 0.5, 128),
    (8, 0.6, 128),

    (13, 0.4, 128),
    (13, 0.5, 128),
    (13, 0.6, 128),

    (5, 0.5, 256),
    (5, 0.5, 512),
    (5, 0.5, 1024),

]

TOKENIZER_PATTERN = r"[ ,.?!:;()\"-]|\n|\\"  # tokenizer pattern used in Tulu-3 decont pipeline, NOTE: this might not work well with Chinese text!!

def custom_tokenize(text):
    # We implement this custom-tokenize func
    # to exactly match the tokenization used in
    # Tulu-3 decontamination ES-pipeline.
    # see https://github.com/allenai/open-instruct/blob/4ce14fe5506c3e900214805a375ae1c204edee4e/decontamination/index.py#L10-L32
    # However, note that the Tulu-3 tokenizer
    # does not work well for Chinese, this is
    # a limitation we have to be aware of!!
    text = text.lower()
    tokens = re.split(TOKENIZER_PATTERN, text)
    return [t for t in tokens if t]

def batch_compute_minhashes(docs, n=8, num_perm=128):
    # this function uses 8-grams by default as that's what's used in Tulu-3 (sec 3.2 https://arxiv.org/pdf/2411.15124)
    all_ngrams_encoded = []
    all_texts_encoded = []
    total = 0
    for text in docs:
        total += 1
        tokens = custom_tokenize(text)
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if len(ngrams) < 1:
            # print("Warning: Skipping sample! You have set too high n={} threshold, cannot create n-gram sequence for this sample: {}\n\n\n\n".format(n, text))
            continue
        encoded = [ng.encode('utf-8') for ng in ngrams]
        assert len(encoded) >= 1, "There is something wrong, the number of elements in this sample's n-gram set is less than 1, sample: {}".format(text)
        all_ngrams_encoded.append(encoded)
        all_texts_encoded.append(text)
    return MinHash.bulk(all_ngrams_encoded, num_perm=num_perm), all_texts_encoded

def load_bbh():
    assert "BBH" in os.listdir(DATA_ROOT)
    assert "data" in os.listdir(os.path.join(DATA_ROOT, "BBH"))

    all_jsons = [os.path.join(DATA_ROOT, "BBH", "data", x) for x in os.listdir(os.path.join(DATA_ROOT, "BBH", "data")) if x.endswith(".json")]
    data_dict = {}
    for j in all_jsons:
        with open(j, 'r') as f:
            examples = json.load(f)["examples"]
            # extract only the prompts to build the index, as done in Tulu-3
            inps = [x["input"] for x in examples]
            data_dict["bbh/"+os.path.basename(j).removesuffix(".json")] = inps

    return data_dict

def load_gaokao():
    assert "GAOKAO-BENCH" in os.listdir(DATA_ROOT)
    assert "data" in os.listdir(os.path.join(DATA_ROOT, "GAOKAO-BENCH"))

    all_jsons = [x for x in glob(os.path.join(DATA_ROOT, "GAOKAO-BENCH", "**", "*.json"), recursive=True)]

    data_dict = {}
    for j in all_jsons:
        with open(j, 'r') as f:
            examples = json.load(f)["example"]
            # extract only the prompts to build the index, as done in Tulu-3
            inps = [x["question"] for x in examples]
            data_dict["gaokao/"+os.path.basename(j).removesuffix(".json")] = inps

    return data_dict

def load_theoremqa():
    assert "TheoremQA" in os.listdir(DATA_ROOT)
    assert "test.csv" in os.listdir(os.path.join(DATA_ROOT, "TheoremQA"))

    df = pd.read_csv(os.path.join(DATA_ROOT, "TheoremQA", "test.csv"))
    # extract only the prompts to build the index, as done in Tulu-3
    questions = df["Question"].tolist()

    return {"theoremqa": questions}

def load_ceval():
    assert "ceval" in os.listdir(DATA_ROOT)
    assert "formal_ceval" in os.listdir(os.path.join(DATA_ROOT, "ceval"))

    all_csvs = [x for x in glob(os.path.join(DATA_ROOT, "ceval", "formal_ceval", "test", "**", "*.csv"), recursive=True)]

    data_dict = {}
    for j in all_csvs:
        df_ = pd.read_csv(j)
        # extract only the prompts to build the index, as done in Tulu-3
        qs = df_["question"].tolist()
        data_dict["ceval/"+os.path.basename(j).removesuffix(".csv")] = qs

    return data_dict

def load_cmmlu():
    assert "cmmlu" in os.listdir(DATA_ROOT)
    assert "test" in os.listdir(os.path.join(DATA_ROOT, "cmmlu"))

    all_csvs = [x for x in glob(os.path.join(DATA_ROOT, "cmmlu", "test", "**", "*.csv"), recursive=True)]

    data_dict = {}
    for j in all_csvs:
        df = pd.read_csv(j)
        # extract only the prompts to build the index, as done in Tulu-3
        qs = df["Question"].tolist()
        data_dict["cmmlu/"+os.path.basename(j).removesuffix(".csv")] = qs

    return data_dict

def load_mmlu():
    assert "mmlu" in os.listdir(DATA_ROOT)
    assert "test" in os.listdir(os.path.join(DATA_ROOT, "mmlu"))

    all_csvs = [x for x in glob(os.path.join(DATA_ROOT, "mmlu", "test", "**", "*.csv"), recursive=True)]

    data_dict = {}
    for j in all_csvs:
        df = pd.read_csv(j, header=None)
        # extract only the prompts to build the index, as done in Tulu-3
        qs = df[0].tolist()
        data_dict["mmlu/"+os.path.basename(j).removesuffix(".csv")] = qs

    return data_dict

def load_gsm8k():
    assert "gsm8k" in os.listdir(DATA_ROOT)
    assert "test.jsonl" in os.listdir(os.path.join(DATA_ROOT, "gsm8k"))

    with open(os.path.join(DATA_ROOT, "gsm8k", "test.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3
        questions = [json.loads(x)["question"] for x in f]

    return {"gsm8k": questions}

def load_hellaswag():
    assert "hellaswag" in os.listdir(DATA_ROOT)
    assert "hellaswag.jsonl" in os.listdir(os.path.join(DATA_ROOT, "hellaswag"))

    with open(os.path.join(DATA_ROOT, "hellaswag", "hellaswag.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3
        questions = [json.loads(x)["query"] for x in f]

    return {"hellaswag": questions}

def load_humaneval():
    assert "humaneval" in os.listdir(DATA_ROOT)
    assert "human-eval-v2-20210705.jsonl" in os.listdir(os.path.join(DATA_ROOT, "humaneval"))

    with open(os.path.join(DATA_ROOT, "humaneval", "human-eval-v2-20210705.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3        
        questions = [json.loads(x)["prompt"] for x in f]

    return {"humaneval": questions}

def load_math():
    assert "math" in os.listdir(DATA_ROOT)
    assert "math.json" in os.listdir(os.path.join(DATA_ROOT, "math"))

    with open(os.path.join(DATA_ROOT, "math", "math.json"), "r") as f:
        examples = json.load(f)

    # extract only the prompts to build the index, as done in Tulu-3
    questions = [v["problem"] for v in examples.values()]

    return {"math": questions}

def load_mbpp():
    assert "mbpp" in os.listdir(DATA_ROOT)
    assert "mbpp.jsonl" in os.listdir(os.path.join(DATA_ROOT, "mbpp"))

    with open(os.path.join(DATA_ROOT, "mbpp", "mbpp.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3
        questions = [json.loads(x)["text"] for x in f]

    return {"mbpp": questions}

def load_mbpp_cn():
    assert "mbpp_cn" in os.listdir(DATA_ROOT)
    assert "mbpp_cn.jsonl" in os.listdir(os.path.join(DATA_ROOT, "mbpp_cn"))

    with open(os.path.join(DATA_ROOT, "mbpp_cn", "mbpp_cn.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3
        questions = [json.loads(x)["text"] for x in f]

    return {"mbpp_cn": questions}

def load_naturalquestions():
    assert "nq" in os.listdir(DATA_ROOT)
    assert "nq-test.qa.csv" in os.listdir(os.path.join(DATA_ROOT, "nq"))

    df = pd.read_csv(os.path.join(DATA_ROOT, "nq", "nq-test.qa.csv"), delimiter='\t', header=None)

    # extract only the prompts to build the index, as done in Tulu-3
    qs = df[0].tolist()

    return {"naturalquestions": qs}

def load_race_high():
    assert "race" in os.listdir(DATA_ROOT)
    assert "test" in os.listdir(os.path.join(DATA_ROOT, "race"))

    with open(os.path.join(DATA_ROOT, "race", "test", "high.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3
        questions = [json.loads(x)["question"] for x in f]

    return {"race": questions}

def load_triviaqa():
    assert "triviaqa" in os.listdir(DATA_ROOT)
    assert "trivia-test.qa.csv" in os.listdir(os.path.join(DATA_ROOT, "triviaqa"))

    df = pd.read_csv(os.path.join(DATA_ROOT, "triviaqa", "trivia-test.qa.csv"), delimiter='\t', header=None)

    # extract only the prompts to build the index, as done in Tulu-3
    qs = df[0].tolist()

    return {"triviaqa": qs}

def load_winogrande():
    assert "winogrande" in os.listdir(DATA_ROOT)
    assert "dev.jsonl" in os.listdir(os.path.join(DATA_ROOT, "winogrande"))

    # we use the dev set since that seems to be the one used by both internvl3 and dclm (appendix G.1 https://arxiv.org/pdf/2406.11794)
    with open(os.path.join(DATA_ROOT, "winogrande", "dev.jsonl"), "r") as f:
        # extract only the prompts to build the index, as done in Tulu-3
        questions = [json.loads(x)["sentence"] for x in f]

    return {"winogrande": questions}

def generate_random_text_dataset(num_samples=1_000_000, min_len=5, max_len=20):
    """
    Generate random synthetic text samples for stress testing.
    """
    vocab = [chr(i) for i in range(97, 123)]  # a-z
    dataset = []
    for _ in range(num_samples):
        length = random.randint(min_len, max_len)
        sample = ' '.join(random.choices(vocab, k=length))
        dataset.append(sample)
    return dataset

def init_worker(lsh_pickle_path):
    global global_lsh
    with open(lsh_pickle_path, "rb") as f:
        global_lsh = pickle.load(f)

def query_worker(mh):
    global global_lsh
    start = time.time()
    matches = global_lsh.query(mh)
    end = time.time()
    query_time = end - start
    hit = 1 if matches else 0
    return query_time, hit

def compute_matches(index, queries, device, batch_size=128):
    '''
    Performs exact nearest neighbor search over MinHash signatures using Jaccard similarity.

    This function compares each query MinHash signature against a provided index (dictionary of MinHashes),
    computes the pairwise Jaccard similarity using signature matching, and returns:

        - top1_scores_all: the Jaccard similarity score of the closest match for each query
        - top1_indices_all: the index (position) in the index dictionary of the closest match for each query
        - latency: total time (in seconds) taken for the search over all queries

    The search is batched and runs fully on the specified `device` (e.g., 'cuda' or 'cpu').

    Args:
        index (dict): Dictionary mapping keys to MinHash signatures.
        queries (list): List of MinHash query signatures.
        device (str or torch.device): Device to perform computation on.
        batch_size (int, optional): Number of queries to process per batch. Default is 128.

    Returns:
        top1_scores_all (np.ndarray): Array of top Jaccard similarity scores (shape: [num_queries]).
        top1_indices_all (np.ndarray): Array of indices in the index corresponding to top matches (shape: [num_queries]).
        latency (float): Total time taken (in seconds) for computing all matches.
    '''

    # assert the types
    assert isinstance(index, dict)
    assert isinstance(queries, list)

    # get np index
    index_signatures_np = np.stack([v for v in index.values()])  # (N, D)
    index_signatures = torch.from_numpy(index_signatures_np).to(device)  # (N, D)

    num_queries = len(queries)
    top1_indices_all = np.empty(num_queries, dtype=np.int32)
    top1_scores_all = np.empty(num_queries, dtype=np.float32)

    # get similarities
    st = time.time()
    for batch_index in tqdm(range(0, num_queries, batch_size)):

        # get ranges
        end_index = min(batch_index + batch_size, num_queries)
        batch_range = slice(batch_index, end_index)

        # get np bs-index
        rand_sigs_np = np.stack([r.digest() for r in queries[batch_range]])  # (bs, D)
        rand_sigs = torch.from_numpy(rand_sigs_np).to(device)  # (bs, D)

        rand_expand = rand_sigs[:, None, :]  # (bs, 1, D)
        index_expand = index_signatures[None, :, :]  # (1, N, D)

        matches = (rand_expand == index_expand)  # shape: (bs, N, D)

        counts = matches.sum(dim=2)

        similarity_matrix = counts.float() / rand_sigs.shape[1]  # (bs, N)

        top1_scores, top1_indices = similarity_matrix.max(dim=1)  # (bs,)

        # Save results to final arrays
        top1_scores_all[batch_range] = top1_scores.cpu().numpy()
        top1_indices_all[batch_range] = top1_indices.cpu().numpy()

    return top1_scores_all, top1_indices_all, time.time()-st

def test_latency_exact(args, index):

    # First generate a random text dataset
    random_dataset = generate_random_text_dataset(num_samples=1_000_000)

    # Next, get all the signatures
    random_signatures, random_samples_for_matches = batch_compute_minhashes(random_dataset, n=args.n_gram_threshold, num_perm=args.signature_size)

    # Get the matches
    top1_scores, top1_indices, latency = compute_matches(index, random_signatures, args.device, args.batch_size)

    # Compute binary hit vector
    hits = top1_scores >= args.sim_threshold  # shape: (num_queries,)

    # Total number of hits
    total_hits = hits.sum()

    print(f"Exact-search Test Latency Results:")
    print(f"Time for 1M queries: {latency} seconds")
    print(f"Total Queries: {len(random_signatures)}")
    print(f"Total Hits: {total_hits}")

def test_latency(args, lsh):

    # NOTE: [possibly buggy] we haven't tested the LSH variant for correctness fully, might want to do that later if needed, but for now we are sticking to the exact search version.

    # First generate a random text dataset
    random_dataset = generate_random_text_dataset(num_samples=50_000)

    # Next, get all the signatures
    random_signatures = batch_compute_minhashes(random_dataset, n=args.n_gram_threshold, num_perm=args.signature_size)

    # query each signature in the lsh index
    query_times = []
    total_hits = 0

    for sample, mh in tqdm(zip(random_dataset, random_signatures), desc="Querying random dataset against LSH", ascii=True, total=len(random_signatures)):
        start = time.time()
        matches = lsh.query(mh)
        end = time.time()

        query_times.append(end - start)
        if matches:
            total_hits += 1

    avg_latency_ms = np.mean(query_times) * 1000  # convert to milliseconds
    qps = 1.0 / np.mean(query_times)  # queries per second

    print(f"Test Latency Results:")
    print(f"Average Query Latency: {avg_latency_ms:.4f} ms")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print(f"Total Queries: {len(random_signatures)}")
    print(f"Total Hits: {total_hits}")

def test_latency_mp(args, lsh):

    # NOTE: [possibly buggy] we haven't tested the LSH variant for correctness fully, might want to do that later if needed, but for now we are sticking to the exact search version.

    # Save the LSH temporarily to share with workers
    tmp_lsh_path = "temp_lsh.pkl"
    with open(tmp_lsh_path, "wb") as f:
        pickle.dump(lsh, f)

    # First generate a random text dataset
    random_dataset = generate_random_text_dataset(num_samples=50_000)

    # Next, get all the signatures
    random_signatures = batch_compute_minhashes(random_dataset, n=args.n_gram_threshold, num_perm=args.signature_size)

    # Multiprocessing pool
    print("Starting multiprocessing querying...")
    with mp.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(tmp_lsh_path,)
    ) as pool:
        results = list(tqdm(pool.imap(query_worker, random_signatures),
                            total=len(random_signatures),
                            desc="Querying LSH", ascii=True))

    # Cleanup temp LSH file
    os.remove(tmp_lsh_path)

    # Unpack results
    query_times, hits = zip(*results)

    avg_latency_ms = np.mean(query_times) * 1000  # ms
    qps = 1.0 / np.mean(query_times)
    total_hits = sum(hits)

    print(f"Test-MP Latency Results:")
    print(f"Average Query Latency: {avg_latency_ms:.4f} ms")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print(f"Total Queries: {len(random_signatures)}")
    print(f"Total Hits: {total_hits}")

def store_visualizations(top1_scores, top1_indices, query_examples, index_examples, file_to_save):

    # create storage bins
    bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    binned_data = {str(b): [] for b in bins}

    # iterate over all scores and assign to bins
    for i, score in enumerate(top1_scores):
        for b in bins:
            # store in appropriate bins, filter only those samples that have larger than 0.1 match score
            if score <= b and score >= 0.1:
                # Save the query, matched example, and score
                example = {
                    "train": query_examples[i],
                    "test": index_examples[top1_indices[i]],
                    "score": round(float(score), 4)
                }
                binned_data[str(b)].append(example)
                break  # bin found, move to next score

    # save to JSON
    os.makedirs(os.path.dirname(file_to_save), exist_ok=True)
    with open(file_to_save, "w") as f:
        json.dump(binned_data, f, indent=4)

    print(f"Saved visualizations to {file_to_save}")

def get_hits_for_visualization(args, data_dicts):

    # do this visualization for flan, flanv2, mathqa, gsm-8k, dolly, ultrafeedback, ultrachat, slimorca, numina-math-1.5, xcoder_80k

    if args.vis_root:
        os.makedirs(args.vis_root, exist_ok=True)
        os.makedirs(args.vis_root+"/plots", exist_ok=True)
        os.makedirs(args.vis_root+"/visuals", exist_ok=True)
        for tk in TRAIN_ROOTS:
            os.makedirs(args.vis_root+"/visuals/{}".format(tk), exist_ok=True)

    train_dicts = {}

    # iterate over all training sets we want to run tests over
    for train_key in tqdm(TRAIN_ROOTS, desc="Loading train splits...", total=len(TRAIN_ROOTS), ascii=True):

        # open each train jsonl file
        with open(TRAIN_ROOTS[train_key], "r") as f:

            # get full dataset
            arr = [json.loads(x) for x in f]

        # take the first 20_000 prompt samples from each dataset as queries
        p_list = []
        for ind in range(min(20_000, len(arr))):
            # since we do n-gram matches simply concatenate the user parts of the conversation
            p_list.append(" ".join([x["value"] for x in arr[ind]["conversations"] if x["from"]=="human"]))

        train_dicts[train_key] = p_list

    sweep_latency_results = {}
    sweep_hits_results = {}

    # Now do a sweep over the params to get individual hit-rates
    for param in tqdm(PARAM_SWEEPS, total=len(PARAM_SWEEPS), ascii=True, desc="Running param sweeps!"):

        # get individual params
        ng_threshold, sim_threshold, sig_size = param[0], param[1], param[2]

        # construct index for current param set
        index = OrderedDict()
        inds2samples = []
        tot = 0
        for data_key, data_list in tqdm(data_dicts.items(), total=len(data_dicts), ascii=True, desc="Building Index!"):
            curr_hashes, curr_texts = batch_compute_minhashes(data_list, n=ng_threshold, num_perm=sig_size)
            tot += len(data_list)
            for idx, (mh, tx) in enumerate(zip(curr_hashes, curr_texts)):
                key = f"{data_key}-{idx}"
                index[key] = mh.digest()  # copy-safe op rather than accessing hashvalues directly
                inds2samples.append(tx)
        print("Constructed index of size {} for params: {}, skipped {} samples due to truncation".format(len(index), param, tot-len(index)))

        # store all results for current sweep
        all_results = []

        sweep_hits_results[param] = {}
        sweep_latency_results[param] = {}

        # construct output file path using args
        output_filename = f"results_ngram{ng_threshold}_thresh{sim_threshold}_sigsize{sig_size}.txt"
        output_filepath = os.path.join(args.vis_root, output_filename)

        # get minhash signatures for all the query samples in the corresponding train sets
        for train_data_key, train_data_list in train_dicts.items():

            # get current signatures
            train_signatures, train_samples_for_matches = batch_compute_minhashes(train_data_list, n=ng_threshold, num_perm=sig_size)

            # get the matches
            if sig_size < 256:
                top1_scores, top1_indices, latency = compute_matches(index, train_signatures, args.device, 128)
            else:
                top1_scores, top1_indices, latency = compute_matches(index, train_signatures, args.device, 16)

            # save visualizations if not already stored
            visuals_folder = f"ngram{ng_threshold}_sigsize{sig_size}"
            visuals_folder = os.path.join(args.vis_root, "visuals", train_data_key, visuals_folder)
            visuals_filepath = os.path.join(visuals_folder, "visualization.json")

            if not os.path.exists(visuals_filepath):
                store_visualizations(top1_scores, top1_indices, train_samples_for_matches, inds2samples, visuals_filepath)

            # compute binary hit vector
            hits = top1_scores >= sim_threshold  # shape: (num_queries,)

            # indices of queries that had a hit
            query_indices_with_hits = np.where(hits)[0]  # shape: (num_hits,)

            # corresponding indices in the index for those queries
            index_matches_for_hits = top1_indices[hits]  # shape: (num_hits,)

            # corresponding scores in the index for those queries
            score_matches_for_hits = top1_scores[hits]  # shape: (num_hits,)

            # total number of hits
            total_hits = hits.sum()

            result_text = (
                f"Results for {train_data_key}:\n"
                f"Time for queries: {latency:.2f} seconds\n"
                f"Total Queries: {len(train_signatures)}\n"
                f"Total Hits: {total_hits} ({(total_hits / len(train_signatures)) * 100:.2f}%)\n"
                + "___________" * 5 + "\n"
            )

            # Print and store
            print(result_text, end="")
            all_results.append(result_text)
            sweep_hits_results[param][train_data_key] = float(total_hits * 100. / len(train_signatures))
            sweep_latency_results[param][train_data_key] = float(latency)

        # write all results to file at once
        if args.vis_root:
            with open(output_filepath, "w") as f:
                f.writelines(all_results)

    with open(os.path.join(args.vis_root, "sweep_hits_results.pickle"), "wb") as f:
        pickle.dump(sweep_hits_results, f)
    with open(os.path.join(args.vis_root, "sweep_latency_results.pickle"), "wb") as f:
        pickle.dump(sweep_latency_results, f)

def main(args):

    # Seed libs that we use
    random.seed(0)

    # First, load all test datasets (should be possible since they are <200MB on disk)
    print("Loading datasets...")
    data_dicts = {}
    data_dicts.update(load_bbh())
    data_dicts.update(load_gaokao())
    data_dicts.update(load_theoremqa())
    data_dicts.update(load_ceval())
    data_dicts.update(load_cmmlu())
    data_dicts.update(load_gsm8k())
    data_dicts.update(load_hellaswag())
    data_dicts.update(load_humaneval())
    data_dicts.update(load_math())
    data_dicts.update(load_mbpp())
    data_dicts.update(load_mbpp_cn())
    data_dicts.update(load_mmlu())
    data_dicts.update(load_naturalquestions())
    data_dicts.update(load_race_high())
    data_dicts.update(load_triviaqa())
    data_dicts.update(load_winogrande())

    # Next, create the index, whether it is a MinHashLSH index or an exact match index (i.e. directly storing the hashes)
    # for reference:
    # gpt-3 used 10 sized signature
    # lee et al. uses 5-grams and 9000 sized signature, 0.8 as threshold 
    # gopher used 13-grams with 0.8 threshold, they used same params as lee et al. otherwise
    # refined web uses the same setup as lee et al
    # pile uses 10 sized signature with 0.5 threshold
    # dclm uses 1395 sized signature with 5-grams
    # fineweb uses 5-grams and 112 sized signatures
    # NOTE: we've only fully tested the exact-search variant, the LSH variant needs further testing, but for now we are sticking to using the exact-search variant.
    if args.do_LSH:
        index = MinHashLSH(num_perm=128, params=(32, 4))
    else:
        index = OrderedDict()

    # store the exact samples with the keys / indices for easy retrieval later
    keys2samples = OrderedDict()
    inds2samples = []

    # Then, create individual minhashes for all test-samples and insert into index
    tot = 0
    for data_key, data_list in tqdm(data_dicts.items(), total=len(data_dicts), ascii=True, desc="Building Index!"):
        curr_hashes, curr_texts = batch_compute_minhashes(data_list, n=args.n_gram_threshold, num_perm=args.signature_size)
        tot += len(data_list)
        for idx, (mh, tx) in enumerate(zip(curr_hashes, curr_texts)):
            key = f"{data_key}-{idx}"
            if args.do_LSH:
                index.insert(key, mh)
            else:
                index[key] = mh.digest()  # copy-safe op rather than accessing hashvalues directly
            keys2samples[key] = tx
            inds2samples.append(tx)
    itype = "exact" if not args.do_LSH else "LSH"
    ilen = len(index) if not args.do_LSH else len(index.keys)
    assert ilen == len(keys2samples)
    assert ilen == len(inds2samples)
    print(f"Built {itype} index with {ilen} entries. While constructing index, skipped {tot-len(index)} samples due to truncation.")

    # Save index and mapping
    assert args.index_path.endswith(".index"), "index name should end with .index"
    with open(args.index_path, 'wb') as f:
        pickle.dump(index, f)
    map_path_ = args.index_path.removesuffix(".index")+".keys2samples"
    with open(map_path_, 'wb') as f:
        pickle.dump(keys2samples, f)
    map_path = args.index_path.removesuffix(".index")+".inds2samples"
    with open(map_path, 'wb') as f:
        pickle.dump(inds2samples, f)
    print(f"Saved {itype} index to {args.index_path} and mappings to {map_path_} and {map_path}")

    if args.do_tests:

        # Running tests for latency
        if args.do_LSH:
            test_latency(args, index)
            test_latency_mp(args, index)
        else:
            test_latency_exact(args, index)

        # Running tests with train splits
        if not args.do_LSH:
            # NOTE: for now we are just implementing support for the exact search, if needed we can do it also for LSH later!
            get_hits_for_visualization(args, data_dicts)

if __name__ == "__main__":

    # To run this script:
    # python build_text_test_index.py --index_path test_texts_minhash.index --vis_root text_decontamination_results

    parser = ArgumentParser()
    parser.add_argument('--index_path', type=str, required=True, 
                        help="Where to save the MinHash-Exact/LSH index.")
    parser.add_argument('--do_LSH', action="store_true", help="whether to construct an LSH index or an exact-matching index.")
    parser.add_argument('--do_tests', action="store_true", help="whether to run tests or not.")
    parser.add_argument('--vis_root', type=str, default=None, 
                        help="Will qualitatively display some matches here if set.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_gram_threshold', type=int, default=5)
    parser.add_argument('--sim_threshold', type=float, default=0.55)
    parser.add_argument('--signature_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    if args.do_LSH:
        assert 'cuda' not in args.device, 'for LSH matching, we cannot use GPU accelerators!!'

    # this function will both build the index and run some tests
    main(args)


