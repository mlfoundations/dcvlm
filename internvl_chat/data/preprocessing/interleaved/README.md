# Download Instructions for Large Multimodal Interleaved / Image-Text Datasets

These are download instructions for the larger multimodal interleaved and image-text datasets in our pool: MINT-1T-HTML, OmniCorpus-CC, WanJuan1.0, GRIT-Caption and SA-1B (used by AS-V1 and InternVL-SA-1B-Caption).

All SLURM launchers read your account and partitions from `internvl_chat/.env` (`ACCOUNT`, `PARTITION`, `INTERNET_PARTITION`; see `internvl_chat/.env.template`). Downloads are submitted to `INTERNET_PARTITION`, since they need internet access.
Before submitting, activate your own environment (with `img2dataset` installed) in the per-job scripts `download_*.sh`, where marked with `# NOTE: activate your own environment here`.

## MINT-1T-HTML / OmniCorpus-CC

For each of these datasets there are four files:
- `download_mint_parquets.py` / `download_omnicc_parquets.py`
- `download_mint.py` / `download_omnicc.py`
- `download_mint.sh` / `download_omnicc.sh`
- `download_multi_mint.sh` / `download_multi_omnicc.sh`

1) The `download_*_parquets.py` scripts download the metadata parquet files from Hugging Face, which contain the image URLs and the texts in interleaved sequence format:
```bash
# for mint-html
python download_mint_parquets.py

# for omnicc
python download_omnicc_parquets.py
```
Change the `output_dir` variable in the script to download to your location of choice.

2) Once the parquets are downloaded, download the images into WebDataset shards. The parquet files are shuffled with a fixed seed and the slice `[START, END)` is downloaded, so you can split the work across several submissions (e.g. `0-200`, `200-400`, ...):
```bash
# for mint-html
PARQUET_ROOT=/path/to/mint/parquets OUTPUT_DIR=/path/to/mint START=0 END=200 ./download_multi_mint.sh

# for omnicc
PARQUET_ROOT=/path/to/omnicc/parquets OUTPUT_DIR=/path/to/omnicc START=0 END=200 ./download_multi_omnicc.sh
```
Each launcher submits a chain of dependent jobs, so a download that hits the walltime is resumed by the next job.

## WanJuan1.0

```bash
URL_LIST=/path/to/WanJuan1.0/multi-modal/part-XXXX.jsonl.tar.gz OUTPUT_FOLDER=/path/to/wanjuan/shards ./download_multi_wanjuan.sh
```

## GRIT-Caption

```bash
python download_grit_caption_parquets.py --output_dir /path/to/grit/parquets
PARQUETS_DIR=/path/to/grit/parquets/grit-20m SHARDS_DIR=/path/to/grit/shards ./download_multi_grit_caption.sh
```

## SA-1B (for AS-V1 and InternVL-SA-1B-Caption)

1) Request access to SA-1B from [Meta](https://ai.meta.com/datasets/segment-anything-downloads/) and save the links file you receive (a tsv with `file_name` and `cdn_link` columns).

2) Download and untar the images:
```bash
LINKS_FILE=/path/to/sa1b_links.tsv OUTPUT_DIR=/path/to/sa1b/tars ./sa1b/download_multi_sa1b.sh
TARS_DIR=/path/to/sa1b/tars OUTDIR=/path/to/sa1b_raw ./sa1b/untar_multi_sa1b.sh
```

3) Build the shards:
```bash
# AS-V1
AS_ANNOTATIONS_DIR=/path/to/AS-100M SA1B_IMAGES_DIR=/path/to/sa1b_raw AS_OUT_DIR=/path/to/as100m/shards python as_v1.py

# InternVL-SA-1B-Caption
python internvl_sa1b_caption.py --annotations_dir /path/to/InternVL-SA-1B-Caption --images_root /path/to/sa1b_raw --output_dir /path/to/internvl_sa1b_caption/shards
```

## Resharding

`reshard_mint_pdf.py` and `reshard_multimodal_textbook.py` convert the original MINT-1T-PDF and Multimodal-Textbook releases into our shard format; see `--help` for their arguments.
