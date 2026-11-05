#!/bin/bash

# NOTE: activate your own environment here (needs img2dataset), e.g.:
# source /path/to/your/env/bin/activate

: "${PARQUETS_DIR:?set PARQUETS_DIR to the downloaded grit-20m parquets}"
: "${SHARDS_DIR:?set SHARDS_DIR to the output folder}"

mkdir -p $SHARDS_DIR
echo "Shards will be saved in $SHARDS_DIR"

img2dataset --url_list "$PARQUETS_DIR" --input_format "parquet" \
    --url_col "url" --caption_col "caption" --output_format webdataset \
    --output_folder "$SHARDS_DIR" --processes_count 16 --thread_count 64 --image_size 256 \
    --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True --number_sample_per_shard 10_000 \
    --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
    --enable_wandb True
