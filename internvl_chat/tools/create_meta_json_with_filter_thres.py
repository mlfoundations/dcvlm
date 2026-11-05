import os
import json

input_path = os.path.join(os.environ.get("META_DIR", "/path/to/meta"), "full_run_meta_small_pool.json")
data_folder = os.path.join(os.environ.get("DATA_ROOT", "/path/to/pools"), "dmt_small/")
filters = ["length_tokens"]
# datasets that are split into multiple folders (e.g. "mint_html_vishaal", "mint_html_matteo")
special_datasets = ["mint_html", "omnicc", "relaion", "wanjuan"]

for ptl in [0.0, 0.1, 0.3, 0.5, 0.9]:
    if isinstance(ptl, float):
        output_path = input_path.replace(".json", "_" + "_".join(filters) + f"_f{ptl}.json")
    else: # if ptl is a tuple of (lower percentile, upper percentile)
        output_path = input_path.replace(".json", "_" + "_".join(filters) + f"_f{ptl[0]}_f{ptl[1]}.json")
    print(output_path)

    with open(input_path, 'r') as fin:
        metadata = json.load(fin)
        for dataset in metadata.keys():
            metadata[dataset]["filters"] = {}

            if any([dataset.startswith(prefix) for prefix in special_datasets]):
                if dataset == "relaion_2b":
                    prefix = "relaion"
                else:
                    prefix = dataset
                all_related_datasets = [folder for folder in os.listdir(data_folder) if folder.startswith(prefix)]
                print(prefix, all_related_datasets)
                filter_threshold_metadata = []
                for related_dataset in all_related_datasets: # get all dataset splits with the same prefix (e.g. "mint_html")
                    with open(os.path.join(data_folder, related_dataset, "extracted_percentiles.json"), 'r') as f_threshold:
                        filter_threshold_metadata.append(json.load(f_threshold))

            else:
                with open(os.path.join(data_folder, dataset, "extracted_percentiles.json"), 'r') as f_threshold:
                    filter_threshold_metadata = [json.load(f_threshold)]

            for filter in filters:
                if any([filter not in threshold_metadata for threshold_metadata in filter_threshold_metadata]):
                    print(f"----------{filter} is not present in {dataset}----------")
                    continue
                else:
                    if isinstance(ptl, float): # filter with only a lower threshold
                        lower_thres = ptl
                        upper_thres = None
                    else: # filter with both lower and upper thresholds
                        lower_thres = ptl[0]
                        upper_thres = ptl[1]

                    if lower_thres == 0.0:
                        metadata[dataset]["filters"][filter] = [{"threshold": 0.0, "sign": ">="}]
                    else:
                        # compute the average threshold using the percentile stats from each of the dataset splits
                        avg_threshold = sum([threshold_metadata[filter][str(lower_thres)] for threshold_metadata in filter_threshold_metadata])/len(filter_threshold_metadata)
                        metadata[dataset]["filters"][filter] = [{"threshold": avg_threshold, "sign": ">="}]
                    
                    if upper_thres is not None:
                        # compute the average threshold using the percentile stats from each of the dataset splits
                        avg_threshold = sum([threshold_metadata[filter][str(upper_thres)] for threshold_metadata in filter_threshold_metadata])/len(filter_threshold_metadata)
                        metadata[dataset]["filters"][filter].append({"threshold": avg_threshold, "sign": "<"})

    with open(output_path, 'w') as fout:
        json.dump(metadata, fout, indent=4)
