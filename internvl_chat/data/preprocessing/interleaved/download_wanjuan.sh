#!/bin/bash

# NOTE: activate your own environment here (needs img2dataset), e.g.:
# source /path/to/your/env/bin/activate

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT # necessary

python download_wanjuan.py --url_list "$URL_LIST" --output_folder "$OUTPUT_FOLDER" &
wait
