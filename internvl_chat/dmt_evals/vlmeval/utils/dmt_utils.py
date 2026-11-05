import os
import os.path as osp
import warnings

from vlmeval.smp import listinstr, load, is_valid_storage
from vlmeval.dataset import DMT_SUPPORTED_DATASETS


DEFAULT_CKPT_DIR = "../training_outputs"


def get_ckpt_code(pretrained_path, wrap_tag="<ckpt>"):
    ckpt_abspath = osp.abspath(pretrained_path)
    ckpt_root = osp.abspath(os.environ.get("CKPT_DIR", DEFAULT_CKPT_DIR))
    if ckpt_root in ckpt_abspath:
        ckpt_code = ckpt_abspath.replace(ckpt_root + "/", "").replace("/", "__")
    else:
        ckpt_code = pretrained_path.replace("/", "__")
    return f"{wrap_tag}{ckpt_code}{wrap_tag}"


def get_texteval_score(
    model: str,
    dataset: str,
    ckpt: str,
    eval_root: str,
    core_metric_only: bool = False,
    suppress_warnings: bool = False,
):
    folder = f"{eval_root}/text_evals/{model}/{ckpt}/{dataset}"
    subfolders = os.listdir(folder)
    # subfolders follow a timestamp, which is lexicographically sorted
    subfolder = sorted(subfolders, reverse=True)[0]
    # each subfolder has a "summary" folder with the results
    summary = osp.join(folder, subfolder, "summary")
    assert osp.isdir(summary), f"Summary folder not found: {summary}"
    # also summaries have timestamps (especially useful with --reuse)
    csv_file = sorted([f for f in os.listdir(summary) if f.endswith(".csv")], reverse=True)[0]
    csv_timestamp = osp.splitext(csv_file)[0].replace("summary_", "")
    if csv_timestamp != subfolder:
        warnings.warn(
            f"CSV file timestamp {csv_timestamp} does not match subfolder {subfolder}"
            f" If you launched opencompass with --reuse, this is ok. Otherwise, check."
            f" The dataset is {dataset}. Reference folder: {summary}"
        )
    csv_file = osp.join(summary, csv_file)
    assert osp.isfile(csv_file), f"CSV file not found: {csv_file}"
    data = load(csv_file)
    res = {}
    column = f"{model}-{ckpt}"
    # now the boring if-else according to the dataset
    if listinstr([
        "c3", "gsm8k", "hellaswag", 
        "nq", "theoremqa", "triviaqa", 
        "winogrande", "math", "humaneval"], 
    dataset):
        # these only have one line
        res[dataset] = float(data[column].iloc[0])
    elif listinstr(["cmmlu", "ceval", "mmlu", "race"], dataset):
        # these have many subtasks, and all are valid
        all_results = {}
        for subtask, result in zip(data["dataset"], data[column]):
            # strip the dataset name from the subtask
            if dataset == "mmlu":
                subtask = subtask.replace("lukaemon_mmlu_", "")
            else:
                subtask = subtask.replace(f"{dataset}-", "")
            all_results[subtask] = float(result)
        # compute task average
        res[dataset + " (task-avg)"] = sum(all_results.values()) / len(all_results)
        # also insert subtask results if needed
        if not core_metric_only:
            res.update(all_results)
    elif listinstr(["gaokao"], dataset):
        # gaokao has some cloze evals which we don't support
        all_results = {}
        for subtask, result in zip(data["dataset"], data[column]):
            if "cloze" in subtask.lower():
                continue
            subtask = "_".join(subtask.split("_")[2:])
            all_results[subtask] = float(result)
        # compute task average
        res[dataset + " (task-avg)"] = sum(all_results.values()) / len(all_results)
        if not core_metric_only:
            res.update(all_results)
    else:
        raise NotImplementedError(f"TextEval dataset {dataset} not implemented.")
    
    for k, v in res.items():
        res[k] = float(v)
    return res




def get_score(
    model: str,
    dataset: str,
    ckpt: str,
    eval_root: str,
    core_metric_only: bool = False,
    suppress_warnings: bool = False,
    use_judge: bool = False,
):
    
    if (
        dataset in DMT_SUPPORTED_DATASETS["text"]
        or dataset in DMT_SUPPORTED_DATASETS["text_math"]
    ):
        return get_texteval_score(
            model,
            dataset,
            ckpt,
            eval_root,
            core_metric_only,
            suppress_warnings
        )

    ckpt_code = get_ckpt_code(ckpt)
    file_name = f"{eval_root}/{model}/{model}_{ckpt_code}_{dataset}"

    # results are dumped on disk differently according to the benchmark, gotta if-else it
    if listinstr(
        [
            "CCBench",
            "MMBench",
            "SEEDBench",
            "MMMU",
            "ScienceQA",
            "TextVQA",
            "AI2D_TEST",
            "MMStar",
            "RealWorldQA",
            "BLINK",
            "VisOnlyQA-VLMEvalKit",
            "GQA",
            "MMT-Bench",
            "DocVQA",
            "ChartQA",
            "InfoVQA",
            "MUIRBench",
            "MMMB",
            "VizWiz",
            "OCRVQA",
            "3DSR",
            "NaturalBench",
            "VMCBench",
            "CV-Bench",
        ],
        dataset,
    ):
        file_suffix = "_acc.csv"
    elif listinstr(["MME-RealWorld"], dataset):
        file_suffix = "_rating.json"
    elif listinstr(
        [
            "MME",
            "Hallusion",
            "LLaVABench",
            "POPE",
            "AMBER",
            "RefCOCO",
            "MathVista",
            "MMVet",
            "MathVision",
            "WildVision",
            "LogicVista",
            "CharXiv",
            "DynaMath",
            "WeMath",
            "MathVerse",
        ],
        dataset,
    ):
        file_suffix = "_score.csv"
    elif listinstr(["COCO", "OCRBench"], dataset):
        file_suffix = "_score.json"
    elif listinstr(["MTVQA", "MMVP"], dataset):
        file_suffix = "_acc.json"
    elif listinstr(["VCR"], dataset):
        if not suppress_warnings:
            warnings.warn("For VCR, we only support the 'VCR_EN_EASY_ALL' currently!")
        file_suffix = "en_easy_score.json"
    elif listinstr(["OlympiadBench"], dataset):
        file_suffix = "_score_result.csv"
    elif listinstr(["CRPE"], dataset):
        file_suffix = "_score.json"
    elif listinstr(["MantisEval"], dataset):
        file_suffix = "_results.json"
    elif listinstr(["VLGuard"], dataset):
        file_suffix = "_overall.json"
    elif listinstr(["MLLMGuard"], dataset):
        file_suffix = "_par.csv"
    elif listinstr(["MMSafetyBench"], dataset):
        file_suffix = "_asr_all.json"
    else:
        print(f"{dataset} not implemented.")
        raise NotImplementedError

    # set the file-suffix s.t. it includes api calls for judges
    possible_judges = [
        "gpt-4-0125",
        "chatgpt-0125",
        "gpt4o-mini",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4o",
    ]

    # search for existing eval file with judge
    found = False
    for judge in possible_judges:
        judge_file_name = file_name + f"_{judge}{file_suffix}"
        if is_valid_storage(judge_file_name):
            found = True
            break
    # if eval file with judge doesn't exist, search for 'exact_matching' file
    # if also 'exact_matching' doesn't exist, support backward-compatibility
    # if not found and use_judge:
    #     raise ValueError(
    #         f"Cannot find judge file for {model} x {ckpt} x {dataset} combo."
    #     )
    if found and use_judge:
        file_name = judge_file_name
    else:
        judge_file_name = file_name + f"_exact_matching{file_suffix}"
        if is_valid_storage(judge_file_name):
            file_name = judge_file_name
        else:
            file_name = file_name + file_suffix
        if not suppress_warnings:
            warnings.warn(f"No judge file found for {model} x {dataset} x {ckpt}")

    data = load(file_name)
    ret = {}

    # yayyy, let's go
    if dataset == "CCBench":
        ret[dataset] = data["Overall"][0] * 100

    elif dataset == "MMBench":
        for n, a in zip(data["split"], data["Overall"]):
            if n == "dev":
                ret["MMBench_DEV_EN"] = a * 100
            elif n == "test":
                ret["MMBench_TEST_EN"] = a * 100

    elif dataset == "MMBench_CN":
        for n, a in zip(data["split"], data["Overall"]):
            if n == "dev":
                ret["MMBench_DEV_CN"] = a * 100
            elif n == "test":
                ret["MMBench_TEST_CN"] = a * 100

    elif listinstr(
        [
            "SEEDBench",
            "ScienceQA",
            "MMBench",
            "AI2D_TEST",
            "MMStar",
            "RealWorldQA",
            "BLINK",
        ],
        dataset,
    ):
        ret[dataset] = data["Overall"][0] * 100

    elif "VMCBench_DEV" == dataset:
        ret[dataset] = data["Overall"][0]

    elif "MME" == dataset:
        if core_metric_only:
            ret[dataset] = data["perception"][0] + data["reasoning"][0]
        else:
            ret[dataset + " Perception"] = data["perception"][0]
            ret[dataset + " Reasoning"] = data["reasoning"][0]

    elif "MMVet" == dataset:
        data = data[data["Category"] == "Overall"]
        ret[dataset] = float(data.iloc[0]["acc"])

    elif "HallusionBench" == dataset:
        data = data[data["split"] == "Overall"]
        ret[dataset] = (
            data.iloc[0]["aAcc"] + data.iloc[0]["fAcc"] + data.iloc[0]["qAcc"]
        ) / 3
        if not core_metric_only:
            for met in ["aAcc", "qAcc", "fAcc"]:
                ret[dataset + f" ({met})"] = float(data.iloc[0][met])

    elif "MMMU" in dataset:
        data = data[data["split"] == "validation"]
        ret["MMMU (val)"] = float(data.iloc[0]["Overall"]) * 100

    elif "RefCOCO" in dataset:
        for _, row in data.iterrows():
            if "val" in row["dataset"]:
                ret[dataset] = row["p1"] * 100
            elif not core_metric_only:
                split = row["dataset"].split("_")[-1]
                ret[dataset + f" ({split})"] = row["p1"] * 100

    elif listinstr(["MathVista", "LogicVista"], dataset):
        data = data[data["Task&Skill"] == "Overall"]
        ret[dataset] = float(data.iloc[0]["acc"])

    elif "MathVision" in dataset:
        data = data[data["Subject"] == "Overall"]
        ret[dataset] = float(data.iloc[0]["acc"])

    elif "MathVerse" in dataset:
        data = data[data["split"] == "Vision Only"]
        ret[dataset] = float(data.iloc[0]["Overall"])

    elif "DynaMath" in dataset:
        data = data[data["Setting"] == "Average"]
        ret[dataset] = float(data.iloc[0]["Overall"]) * 100

    elif "WeMath" in dataset:
        ret[dataset] = float(data["Score (Loose)"][0].replace("%", ""))

    elif "LLaVABench" in dataset:
        data = data[data["split"] == "overall"].iloc[0]
        ret[dataset] = float(data["Relative Score (main)"])

    elif "OCRBench" in dataset:
        ret[dataset] = data["Final Score"]

    elif dataset == "VisOnlyQA-VLMEvalKit":
        for n, a in zip(data["split"], data["Overall"]):
            ret[f"VisOnlyQA-VLMEvalKit_{n}"] = a * 100

    elif dataset == "COCO_VAL":
        # returning Bleu@4 as the main metric
        if core_metric_only:
            ret[dataset + " CIDEr"] = data["CIDEr"]
        else:
            for i in range(4):
                ret[dataset + f" Bleu_{i+1}"] = data["Bleu"][i]
            ret[dataset + " ROUGE_L"] = data["ROUGE_L"]
            ret[dataset + " CIDEr"] = data["CIDEr"]

    elif listinstr(
        ["GQA", "TextVQA", "DocVQA", "ChartQA", "InfoVQA", "VizWiz", "OCRVQA"], dataset
    ):
        ret[dataset] = float(data["Overall"][0])

    elif listinstr(["MMT-Bench", "MUIRBench", "3DSR", "CharXiv", "CV-Bench"], dataset):
        if listinstr(["MMT-Bench"], dataset):
            data = data[data["split"] == "VAL"]
        ret[dataset] = float(data["Overall"][0]) * 100

    elif dataset == "MTVQA_TEST":
        if core_metric_only:
            ret[dataset] = float(data["Average"])
        else:
            # accuracy per language
            for lan, acc in data.items():
                ret[dataset + f" ({lan})"] = float(acc)

    elif dataset == "MMVP":
        ret[dataset] = data["acc"] * 100

    elif listinstr(["MME-RealWorld"], dataset):
        ret[dataset] = float(data["Overall"]) * 100

    elif listinstr(["VCR"], dataset):
        if core_metric_only:
            ret[dataset] = data["Jaccard"] * 100
        else:
            for k in ["Exact_Match", "Jaccard"]:
                ret[f"{dataset} ({k})"] = data[k] * 100

    elif listinstr(["OlympiadBench"], dataset):
        ret[dataset] = float(data["AVG"][0]) * 100

    elif listinstr(["WildVision"], dataset):
        ret[dataset] = float(data["Win Rate"][0]) * 100

    elif listinstr(["CRPE"], dataset):
        ret[dataset] = float(data["total"]) * 100

    elif listinstr(["AMBER"], dataset):
        ret[dataset] = float(data["Avg ACC"][0])
        if not core_metric_only:
            for k in ["Attribute", "Existence", "Relation"]:
                ret[f"{dataset} ({k})"] = float(data[k][0])

    elif listinstr(["POPE"], dataset):
        f1_scores = {}
        for split in ["adversarial", "popular", "random"]:
            subset_data = data[data["split"] == split]
            f1_scores[f"{dataset} ({split})"] = float(subset_data["Overall"].values[0])

        ret[dataset] = sum(f1_scores.values()) / len(f1_scores)
        if not core_metric_only:
            ret.update(f1_scores)

    elif listinstr(["MMMB"], dataset):
        if not suppress_warnings:
            warnings.warn(
                "MMMB parsing is supported, but only for the overall dataset with all languages!"
            )
        lan_scores = dict(zip(data["DATASET"], data["Overall"]))
        lan_scores = {k: float(v) * 100 for k, v in lan_scores.items()}
        ret[dataset] = sum(lan_scores.values()) / len(lan_scores)
        if not core_metric_only:
            ret.update(lan_scores)

    elif listinstr(["MantisEval"], dataset):
        ret[dataset] = data["overall_acc"] * 100
        if not core_metric_only:
            for k, v in data.items():
                if k != "overall_acc":
                    ret[f"{dataset} ({k})"] = v * 100

    elif listinstr(["NaturalBench"], dataset):
        for i, row in data.iterrows():
            if row["Metric"] == "Acc":
                ret[dataset] = float(row["Score"]) * 100
            elif not core_metric_only:
                ret[f'{dataset} ({row["Metric"]})'] = float(row["Score"]) * 100

    elif listinstr(["VLGuard"], dataset):
        # NOTE: changing to "Attack Failure Rate" (1 - ASR) as the main metric
        # (we want higher is better)
        asr_unsafe = data["unsafes_ASR"]
        asr_safe_unsafe = data["safe_unsafes_ASR"]
        asr_avg = (asr_unsafe + asr_safe_unsafe) / 2
        if core_metric_only:
            ret[dataset] = 100 - asr_avg
        else:
            ret[dataset + " AFR (Unsafes)"] = 100 - asr_unsafe
            ret[dataset + " AFR (Safe -> Unsafes)"] = 100 - asr_safe_unsafe
            ret[dataset + " AFR (Avg)"] = 100 - asr_avg

    elif listinstr(["MLLMGuard"], dataset):
        if core_metric_only:
            ret[dataset] = data["avg"].iloc[0] * 100
        else:
            asd_file = file_name.replace(file_suffix, "_asd.csv")
            asd_data = load(asd_file)
            asd = asd_data["avg"].iloc[0] * 100
            ret[dataset + " ASD (Avg)"] = asd
            par = data["avg"].iloc[0] * 100
            ret[dataset + " PAR (Avg)"] = par

    elif listinstr(["MMSafetyBench"], dataset):

        def avg_split(data, split: str = "TYPO"):
            split_slice = data[split]
            split_asr_scalars = [
                1 - float(subtask_results["attack_rate"])
                for _, subtask_results in split_slice.items()
            ]
            return sum(split_asr_scalars) / len(split_asr_scalars) * 100

        # we always compute average asr for the typography split
        avg_typo_asr = avg_split(data, split="TYPO")
        ret[dataset + " (TYPO Avg)"] = avg_typo_asr

        # if we have them (which will not be the case)
        # we also compute average asr on the SD and SD_TYPO
        if not core_metric_only:
            try:
                avg_sd_asr = avg_split(data, split="SD")
                avg_sd_typo_asr = avg_split(data, split="SD_TYPO")
                avg_asr = (avg_typo_asr + avg_sd_asr + avg_sd_typo_asr) / 3
                ret[dataset + " (SD Avg)"] = avg_sd_asr
                ret[dataset + " (SD_TYPO Avg)"] = avg_sd_typo_asr
                ret[dataset + " (ALL Avg)"] = avg_asr
            except KeyError as e:
                if not suppress_warnings:
                    warnings.warn(
                        f"""
                        Missing key error: {e}. 
                        You might need to evaluate on the SD and SD_TYPO splits as well to compute the overall ASR.
                    """
                    )

    for k, v in ret.items():
        ret[k] = float(v)
    return ret
