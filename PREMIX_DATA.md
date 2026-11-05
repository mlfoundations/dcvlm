# Building a Pre-Mixed Dataset

This guide explains how to turn a data mixture (a Meta-JSON file, see [DATA.md](DATA.md#meta-json-file)) into a ready-to-train dataset: a folder of WebDataset shards that any training code can read, without needing our data pipeline. This is how the pre-mixed datasets in the [README](README.md#pre-mixed-ready-to-train-datasets) were built.

## Overview

The process has two steps:

1. **Dump.** Run our training data loader over the mixture and write every sample it produces to disk, until the token budget is reached. The result has exactly the mixture proportions used in training.
2. **Reshuffle.** Each shard written in step 1 comes from a single data loader worker, which only reads a few sources at a time. A single shard can therefore be dominated by a handful of datasets. This doesn't matter if you train on all shards, but it does if you only use some of them. The reshuffle step mixes samples across all shards, so that every shard looks like a small random sample of the whole dataset.

```
Meta-JSON ──(1. dump)──▶ release/*.tar ──(2. reshuffle)──▶ reshuffle/release_shuffled/*.tar
                         correct mixture,                   correct mixture,
                         uneven shards                      every shard representative
```

All commands below are run from the `internvl_chat/` folder. The launchers submit SLURM jobs for you.

## Before you start

1. Set up `internvl_chat/.env` and `internvl_chat/scripts/slurm/system_setup.sh` as described in [SLURM.md](SLURM.md). In particular, `.env` must define `META_DIR`, `ACCOUNT`, `PARTITION` and `INTERNET_PARTITION` (a CPU partition, used by the checking scripts).
2. Choose an output folder `OUT` with plenty of space: about **3× the final dataset size** is needed while the reshuffle is running.
3. The finalize commands below run on the login node and need the training environment:
   ```bash
   cd internvl_chat
   source scripts/slurm/system_setup.sh train
   ```

None of these steps needs GPU compute. By default the jobs still request whole nodes on `$PARTITION`; set `PARTITION` (and `DUMP_GPUS_PER_NODE`) to use a CPU partition instead.

## Step 1: Dump

**Pick a mixture and a size.** `META_JSON` is the path of your Meta-JSON relative to `$META_DIR`. `SCALE` sets how many tokens to write:

| SCALE | tokens |
|---|---|
| `debug` | 50M (quick test) |
| `small` | 6.25B |
| `medium` | 25B |
| `large_1bmodel` | 50B |
| `large2x_1bmodel` | 100B |
| `xlarge` | 200B |

Set `TOT_TOKENS` instead to use any other budget.

**Launch:**
```bash
OUT=/path/to/my_dataset
SCALE=small META_JSON=my_mixture.json NODES=8 OUT=$OUT ./scripts/slurm/dump/launcher.sh
```
Use `QOS=...` and `TIME=...` to pick a different queue or time limit than the defaults in `.env`.

**Wait for it to finish.** Each process writes a marker file when it is done. There are 4 processes per node, so wait until the count reaches `NODES × 4`:
```bash
ls $OUT/markers/*.done | wc -l
```

**Finalize.** This collects the shards into `$OUT/release/` and writes the dataset metadata:
```bash
python -m internvl.train.data.finalize_release --output_folder $OUT
```

**Check the result.** This decodes a sample of the data and checks the mixture proportions and the sample format:
```bash
OUT=$OUT MAX_SAMPLES=5000 ./scripts/quickruns/check_dump_release.sh
```

## Step 2: Reshuffle

The reshuffle runs in two phases, which must be run one after the other.

**Phase 1 (scatter)** sends every sample to one of `NUM_BUCKETS` buckets at random:
```bash
PHASE=scatter OUT=$OUT NODES=8 NUM_BUCKETS=64 ./scripts/slurm/reshuffle/launcher.sh
ls $OUT/reshuffle/scatter_markers/*.done | wc -l    # wait until this reaches NODES × 4
```

**Phase 2 (gather)** shuffles each bucket and writes the final shards:
```bash
PHASE=gather OUT=$OUT NODES=8 NUM_BUCKETS=64 ./scripts/slurm/reshuffle/launcher.sh
ls $OUT/reshuffle/gather_markers/*.done | wc -l     # wait until this reaches NUM_BUCKETS
```
Use the same `NUM_BUCKETS` and `SEED` in both phases. The number of nodes can differ.

**Finalize.** This checks that no sample was lost or duplicated, and collects the shards into `$OUT/reshuffle/release_shuffled/`:
```bash
python -m internvl.train.data.finalize_reshuffle --root $OUT/reshuffle --orig_release $OUT/release
```

**Clean up.** The scatter phase leaves a temporary full copy of the data. Delete it once finalize has succeeded:
```bash
rm -rf $OUT/reshuffle/scatter
```

## Checking the mixing (optional)

Two scripts measure how well the data is mixed:

- `verify_source_dist.sh` compares how often each source dataset appears against its weight in the Meta-JSON, and saves a plot and a CSV.
- `assess_dispersion.sh` measures how evenly sources are spread across shards. It reports a number called **Deff**: around 1 means every shard looks like the whole dataset, and much larger values mean shards are dominated by a few sources. Expect a large value (hundreds) before the reshuffle and around 1–3 after it.

```bash
OUT=$OUT ./scripts/quickruns/verify_source_dist.sh
OUT=$OUT ./scripts/quickruns/assess_dispersion.sh                                                  # before reshuffle
OUT=$OUT RELEASE_DIR=$OUT/reshuffle/release_shuffled ./scripts/quickruns/assess_dispersion.sh      # after reshuffle
```

These scripts read many shards, so they submit a job to `$INTERNET_PARTITION` rather than running on the login node.

## Using the dataset

The final dataset is in `$OUT/reshuffle/release_shuffled/`:
```
release_shuffled/
├── 000000.tar, 000001.tar, ...
├── dataset_info.json    # number of samples and tokens, samples per data type
└── meta_flat.json       # Meta-JSON pointing at the shards, ready for our trainer
```

To train on it with this codebase, use `meta_flat.json` as your Meta-JSON.

Each sample in a shard contains:
- `conversations.txt`: the conversation, with turns separated by `<EOCL>` and an `<image>` placeholder where each image goes. `txt2conversations` in `internvl/train/data/formatters/conversation.py` turns it back into a list of turns.
- numbered images (`0.jpg`, `1.png`, ...), in the order of the `<image>` placeholders. Text-only samples have no images.
- `ds_name.txt`, `orig_url.txt`, `data_type.txt`, `ntok.txt`, `mdp.txt`: information about where the sample came from. They are not needed for training.

## If a job fails

Every step can be resumed: finished processes leave a marker file and are skipped when you resubmit the same command. Before resubmitting, delete the partial output of the processes that did not finish:

```bash
# dump: remove partial shards of unfinished processes, then resubmit the dump command
for t in "$OUT"/staging/r*-*.tar; do r=$(basename "$t" | sed -E 's/-[0-9]+\.tar$//'); [ -f "$OUT/markers/$r.done" ] || rm -f "$t"; done

# scatter: remove partial output of unfinished processes, then resubmit PHASE=scatter
WORLD_SIZE=<NODES*4>
for r in $(seq 0 $((WORLD_SIZE-1))); do rr=$(printf 'r%05d' $r); \
  [ -f $OUT/reshuffle/scatter_markers/$rr.done ] || rm -f $OUT/reshuffle/scatter/bucket_*/$rr-*.tar; done

# gather: remove partial output of unfinished buckets, then resubmit PHASE=gather
for b in $(seq 0 $((NUM_BUCKETS-1))); do bb=$(printf 'b%04d' $b); \
  [ -f $OUT/reshuffle/gather_markers/$bb.done ] || rm -f $OUT/reshuffle/shuffled/$bb-*.tar; done
```

On large runs (100+ nodes), expect occasional node failures; resubmitting after this cleanup is all that is needed.

## Tips for large datasets

- **Set `NUM_BUCKETS` to the number of gather processes** (`NODES × 4`). Each gather process handles whole buckets, so processes beyond `NUM_BUCKETS` sit idle. For example, for the 200B-token datasets we used 128 nodes and `NUM_BUCKETS=512` for both phases.
- **Memory:** each gather process holds `SHUFFLE_BUFSIZE` samples in memory (default 20000, about 4.6 GB). A larger buffer mixes slightly better.
- **Open files:** each scatter process keeps `NUM_BUCKETS` files open, so `NUM_BUCKETS` must stay below your `ulimit -n`.
- **Disk space:** the original shards, the scatter copy and the shuffled shards all exist at the same time until the cleanup step, which is why you need about 3× the dataset size.

## Settings reference

**Dump** (`scripts/slurm/dump/launcher.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `SCALE` | required | token budget (table above) |
| `META_JSON` | required | mixture file, relative to `$META_DIR` |
| `OUT` | required | output folder |
| `NODES` | 8 | number of nodes (4 processes each) |
| `TOT_TOKENS` | from `SCALE` | custom token budget |
| `SEED` | 40 | random seed of the mixture |
| `SAMPLES_PER_SHARD` | 10000 | samples per shard |
| `PACKED_IMAGES` | 24 | maximum images per sample |
| `DUMP_NUM_WORKERS` | 8 | data loader workers per process |
| `QOS` / `TIME` | from `.env` | queue and time limit |

**Reshuffle** (`scripts/slurm/reshuffle/launcher.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `PHASE` | required | `scatter` or `gather` |
| `OUT` | required | output folder of the dump |
| `NODES` | 64 | number of nodes (4 processes each) |
| `NUM_BUCKETS` | 256 | number of buckets; use 64 for small datasets |
| `SHUFFLE_BUFSIZE` | 20000 | samples kept in memory per gather process |
| `SAMPLES_PER_SHARD` | 10000 | samples per final shard |
| `SEED` | 40 | random seed of the reshuffle |
