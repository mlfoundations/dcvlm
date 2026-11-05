#!/bin/bash

# NOTE: activate your own environment here, e.g.:
# source /path/to/your/env/bin/activate

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT # necessary

python ../download_sa1b.py --links_file "$LINKS_FILE" --output_dir "$OUTPUT_DIR" &
wait
