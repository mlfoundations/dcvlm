#!/bin/bash

# NOTE: activate your own environment here (needs img2dataset), e.g.:
# source /path/to/your/env/bin/activate

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT # necessary

python download_omnicc.py --parquet_root "$PARQUET_ROOT" --output_dir "$OUTPUT_DIR" --start "$START" --end "$END" &
wait
