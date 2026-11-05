# Pre-Mixed Release Cheatsheet

How to materialize a Meta-JSON mixture as a flat, codebase-independent
**WebDataset release** (MAmmoTH-VL-style: per-sample `conversations.txt` + numbered
images), so external users can train on the exact mix without our data pipeline.
This is how the pre-mixed datasets in the [README](README.md#pre-mixed-ready-to-train-datasets) were built.

There are **two stages**:

```
 Meta-JSON ──(Stage 1: DUMP)──▶  biased release  ──(Stage 2: RESHUFFLE)──▶  unbiased release
  10c-5m-15t-70i.json           release/*.tar      scatter → gather        release_shuffled/*.tar
                                (faithful mix,      (2-level MapReduce)     (per-shard representative,
                                 per-shard skewed)                           Deff ≈ 1)
```

- **Stage 1 (dump)** replays the *exact* production data path (`DeterministicRandomMix`
  + the same seed/params as `scripts/slurm/train/run.sh`) and writes every drawn sample
  to disk. The corpus is faithful **in aggregate** (per-source sampling matches the meta
  to r≈0.998), but each 10k-sample shard is one worker's live draw-order window, so
  **per-shard source composition is skewed** (design effect Deff ≈ 600; e.g. one shard
  had a source at 19× its global rate, with only ~26/166 sources present).
- **Stage 2 (reshuffle)** rewrites the shards so each one is an independently
  representative random sample of the whole corpus (Deff → ~1–3), matching how
  MAmmoTH-VL's flat shards behave. Online training never needed this (it averages over
  ~2048 workers every step), but a user streaming a *subset* of shards does.

> **Always run commands from `internvl_chat/`.** The launchers submit via `sbatch`/`srun`.

---

## Prerequisites

1. Training env + `PYTHONPATH` (the SLURM scripts do this internally; needed only for the
   login-side `finalize_*` steps):
   ~~~bash
   cd internvl_chat
   source scripts/slurm/system_setup.sh train
   ~~~
2. `.env` must define `META_DIR`, `ACCOUNT`, `INTERNET_PARTITION` (a CPU partition, used by
   the checking wrappers) and the usual SLURM vars (`PARTITION`, `GPUS_PER_NODE`,
   `CPUS_PER_NODE`, `MEMORY`, `EMAIL`, `SLURM_LOGS_DIR`, `EXCLUDE_NODES`).
3. Pick an output root `OUT` with enough space (≈3× the release size at peak, see 2.6).
   A scratch filesystem is ideal; if it purges old files, verify and move the release in time.
4. The dump/reshuffle need no GPU compute. By default they request whole nodes on
   `$PARTITION`; override `PARTITION`/`ACCOUNT`/`DUMP_GPUS_PER_NODE` to use a CPU partition.

---

## Stage 1 — Dump (Meta-JSON → biased pre-mixed release)

### 1.1 Pick the mixture + scale

- Meta-JSON: e.g. `small-pool/10c-5m-15t-70i.json` (pilot) or
  `large-pool/10c-5m-15t-70i.json` (200B), resolved under `$META_DIR`.
- `SCALE` sets the total token budget (mirrors the train launcher; override with `TOT_TOKENS`):

  | SCALE | tokens |
  |---|---|
  | `debug` | 50 M (smoke) |
  | `small` | 6.25 B |
  | `medium` | 25 B |
  | `large_1bmodel` | 50 B |
  | `large2x_1bmodel` | 100 B |
  | `xlarge_1bmodel` / `xlarge` | 200 B |

### 1.2 Launch the dump

~~~bash
OUT=/path/to/dcvlm_release/xlarge
SCALE=xlarge_1bmodel \
META_JSON=large-pool/10c-5m-15t-70i.json \
NODES=128 OUT=$OUT \
./scripts/slurm/dump/launcher.sh
~~~

`WORLD_SIZE = NODES × NTASKS_PER_NODE` (4 ranks/node). Each rank streams the faithful mix
through 8 dataloader workers and writes its share to `$OUT/staging/` with a per-rank token
budget (`TOT_TOKENS / WORLD_SIZE`).

Use the `QOS` / `TIME` env overrides to pick your queue (e.g. a debug QOS for ≤8-node tests):
~~~bash
QOS=<debug-qos> TIME=00:30:00 SCALE=small META_JSON=small-pool/10c-5m-15t-70i.json \
NODES=8 OUT=/path/to/dcvlm_release/small ./scripts/slurm/dump/launcher.sh
~~~

**Monitor / recover** (dump is idempotent per rank via `markers/r<rank>.done`):
~~~bash
ls $OUT/markers/*.done | wc -l          # target = WORLD_SIZE
tail $SLURM_LOGS_DIR/dcvlm_dump_<SCALE>.out
~~~
If a job times out or a node fails (expect the latter at 128 nodes), delete partial staging tars
for marker-less ranks and resubmit the **same** command (finished ranks skip):
~~~bash
for t in "$OUT"/staging/r*-*.tar; do r=$(basename "$t"|sed -E 's/-[0-9]+\.tar$//'); [ -f "$OUT/markers/$r.done" ]||rm -f "$t"; done
ls "$OUT"/markers | wc -l   # then re-run the launch command
~~~

### 1.3 Finalize (metadata only — login OK)

Verifies all rank markers, `os.rename`s `staging/*.tar` → flat `release/000000.tar…`, and
writes `dataset_info.json` + `meta_flat.json`:
~~~bash
python -m internvl.train.data.finalize_release --output_folder $OUT
~~~

### 1.4 Structural check

~~~bash
python -m internvl.train.data.check_release --release_dir $OUT/release --max_samples 5000 --tol 0.03
# convenience wrapper (submits to $INTERNET_PARTITION; decodes images so not on login):
OUT=$OUT MAX_SAMPLES=5000 ./scripts/quickruns/check_dump_release.sh
~~~
PASS = data_type fractions ≈ .10/.05/.15/.70, formatters succeed, `<image>` count == #imgs,
no duplicate member names. (Use a modest `MAX_SAMPLES` — the checker is single-core PIL decode.)

### 1.5 (Optional) Faithfulness + dispersion diagnostics

~~~bash
# per-source sampling vs meta weight (r≈0.998), plot + CSV:
OUT=$OUT ./scripts/quickruns/verify_source_dist.sh
# per-shard dispersion (design effect) — expect HIGH Deff here (this is the bias reshuffle fixes):
OUT=$OUT ./scripts/quickruns/assess_dispersion.sh
~~~

---

## Stage 2 — Global reshuffle (biased → unbiased release)

Two-level MapReduce. A one-pass "assign each sample a random final shard" would need
~10k open writers → FD blow-up, so:

- **Pass 1 (scatter):** every rank reads a strided slice of `release/*.tar` and routes each
  sample (raw bytes, no decode) to one of `NUM_BUCKETS` buckets **uniformly at random**.
  Source-independent routing ⇒ each bucket is already a representative 1/B subsample.
- **Pass 2 (gather):** each rank owns buckets `b % WORLD_SIZE == rank`, reads its buckets in
  **randomized part order** through a **bounded streaming shuffle buffer**, and writes final
  shards. Buffer RAM is capped at ~`SHUFFLE_BUFSIZE` samples (≈230 KB each).

Everything is lossless raw-byte pass-through; samples keep their unique `__key__`.

### 2.1 Scatter (pass 1)

~~~bash
PHASE=scatter OUT=$OUT NODES=64 ./scripts/slurm/reshuffle/launcher.sh
~~~
Wait for all scatter markers, then continue:
~~~bash
ls $OUT/reshuffle/scatter_markers/r*.done | wc -l    # target = WORLD_SIZE (NODES×4)
~~~

### 2.2 Gather (pass 2)

~~~bash
PHASE=gather OUT=$OUT NODES=64 ./scripts/slurm/reshuffle/launcher.sh
ls $OUT/reshuffle/gather_markers/b*.done | wc -l     # target = NUM_BUCKETS (default 256)
~~~

### 2.3 Finalize (metadata only — login OK)

Verifies all scatter + gather markers, cross-checks **conservation**
(scattered == gathered == original `num_samples`), flat-renames
`shuffled/*.tar` → `release_shuffled/000000.tar…`, and carries the original token /
per-data_type totals forward with `globally_shuffled: true`:
~~~bash
python -m internvl.train.data.finalize_reshuffle --root $OUT/reshuffle --orig_release $OUT/release
~~~

### 2.4 Verify the reshuffle worked

~~~bash
RELEASE_DIR=$OUT/reshuffle/release_shuffled OUT=$OUT ./scripts/quickruns/assess_dispersion.sh
~~~
PASS = median **Deff drops to ~1–3** (from ~600), all sources at `shard_cover = 1.00`, and
the .10/.05/.15/.70 bucket fractions unchanged.

### 2.5 Reclaim space

`scatter/` is a full transient copy (~= release size). Delete it once finalize passes:
~~~bash
rm -rf $OUT/reshuffle/scatter
~~~

### 2.6 Scaling notes (128 nodes / full 24TB)

- **`NUM_BUCKETS` should be ≈ the gather `WORLD_SIZE`** — gather assigns bucket `b` to rank
  `b % WORLD_SIZE`, so if `NUM_BUCKETS < WORLD_SIZE` the surplus ranks sit idle. At **128
  nodes (WORLD_SIZE=512)** the default 256 buckets leaves **half the nodes idle in gather**.
  Two clean options:
  - **`NUM_BUCKETS=512`** for both phases → every rank owns exactly 1 bucket (≈203k
    samples / ~48 GB each). Cost: scatter writes `512 × 512 = 262k` part files.
  - **`NUM_BUCKETS=256`**, scatter at 128 nodes, **gather at 64 nodes** → ~131k part
    files, gather ranks own 1 bucket each (≈406k / ~96 GB). `NUM_BUCKETS` and `SEED` must
    match across the two phases; node count may differ.
- Bucket size is **not** bounded by RAM — gather streams through `SHUFFLE_BUFSIZE` (~4.6 GB/rank
  at 20k). Only `SHUFFLE_BUFSIZE` and part-file count scale with the knobs.
- **FD limit:** scatter opens `NUM_BUCKETS` writers/rank; check `ulimit -n` before raising it.
- **Peak disk ≈ 3× release** (~72 TB for 200B tokens): original `release/` + `scatter/` + `shuffled/`
  coexist until finalize renames `shuffled/` and you delete `scatter/`.
- **Expect node failures at 128 nodes** (`srun --kill-on-bad-exit=1` → one death kills the job).
  Both passes are idempotent; resubmit after the cleanup below.

**Recovery.** scatter is idempotent per rank (`scatter_markers/`), gather per bucket
(`gather_markers/`). A marker-less scatter rank has partial parts spread across *every*
bucket dir, so clean by rank:
~~~bash
WORLD_SIZE=<NODES*4>
# scatter: drop partial parts of any rank that has no .done marker, then resubmit PHASE=scatter
for r in $(seq 0 $((WORLD_SIZE-1))); do rr=$(printf 'r%05d' $r); \
  [ -f $OUT/reshuffle/scatter_markers/$rr.done ] || rm -f $OUT/reshuffle/scatter/bucket_*/$rr-*.tar; done
ls $OUT/reshuffle/scatter_markers/*.done | wc -l    # progress toward WORLD_SIZE

# gather: drop shards of any bucket with no .done marker, then resubmit PHASE=gather
for b in $(seq 0 $((NUM_BUCKETS-1))); do bb=$(printf 'b%04d' $b); \
  [ -f $OUT/reshuffle/gather_markers/$bb.done ] || rm -f $OUT/reshuffle/shuffled/$bb-*.tar; done
ls $OUT/reshuffle/gather_markers/*.done | wc -l     # progress toward NUM_BUCKETS
~~~

---

## Worked recipes

### A. Small pilot end-to-end on a debug queue

~~~bash
cd internvl_chat
OUT=/path/to/dcvlm_release/small

# 1. dump (8 nodes) → finalize → check
QOS=<debug-qos> TIME=00:30:00 SCALE=small META_JSON=small-pool/10c-5m-15t-70i.json \
NODES=8 OUT=$OUT ./scripts/slurm/dump/launcher.sh
python -m internvl.train.data.finalize_release --output_folder $OUT
OUT=$OUT MAX_SAMPLES=5000 ./scripts/quickruns/check_dump_release.sh

# 2. reshuffle (8 nodes, smaller buckets keep part-count tidy at this scale)
OUT=$OUT ./scripts/quickruns/assess_dispersion.sh                                   # before: Deff ~600
QOS=<debug-qos> TIME=00:30:00 PHASE=scatter OUT=$OUT NODES=8 NUM_BUCKETS=64 ./scripts/slurm/reshuffle/launcher.sh
QOS=<debug-qos> TIME=00:30:00 PHASE=gather  OUT=$OUT NODES=8 NUM_BUCKETS=64 ./scripts/slurm/reshuffle/launcher.sh
python -m internvl.train.data.finalize_reshuffle --root $OUT/reshuffle --orig_release $OUT/release
RELEASE_DIR=$OUT/reshuffle/release_shuffled OUT=$OUT ./scripts/quickruns/assess_dispersion.sh   # after: Deff ~3
~~~

### B. Full 200B x-large

~~~bash
OUT=/path/to/dcvlm_release/xlarge

# dump: 128 nodes
TIME=02:00:00 SCALE=xlarge_1bmodel META_JSON=large-pool/10c-5m-15t-70i.json \
NODES=128 OUT=$OUT ./scripts/slurm/dump/launcher.sh
python -m internvl.train.data.finalize_release --output_folder $OUT

# reshuffle: 128 nodes, NUM_BUCKETS=512 so every rank owns exactly 1 bucket in both
# phases (WORLD_SIZE=512)
OUT=$OUT ./scripts/quickruns/assess_dispersion.sh                                    # before: Deff ~600
TIME=02:00:00 PHASE=scatter OUT=$OUT NODES=128 NUM_BUCKETS=512 ./scripts/slurm/reshuffle/launcher.sh
# wait for 512 scatter markers:
ls $OUT/reshuffle/scatter_markers/r*.done | wc -l
TIME=02:00:00 PHASE=gather  OUT=$OUT NODES=128 NUM_BUCKETS=512 ./scripts/slurm/reshuffle/launcher.sh
# wait for 512 gather markers:
ls $OUT/reshuffle/gather_markers/b*.done | wc -l
python -m internvl.train.data.finalize_reshuffle --root $OUT/reshuffle --orig_release $OUT/release
RELEASE_DIR=$OUT/reshuffle/release_shuffled OUT=$OUT ./scripts/quickruns/assess_dispersion.sh   # after: Deff ~1-3
rm -rf $OUT/reshuffle/scatter                                                        # reclaim ~24TB
~~~

---

## Knobs reference

**Dump** (`scripts/slurm/dump/launcher.sh`):

| Var | Default | Meaning |
|---|---|---|
| `SCALE` | — (required) | token budget preset (table above) |
| `META_JSON` | — (required) | path under `$META_DIR` |
| `OUT` | — (required) | output root |
| `NODES` | 8 | → `WORLD_SIZE = NODES × NTASKS_PER_NODE` |
| `NTASKS_PER_NODE` | 4 | ranks per node |
| `TOT_TOKENS` | from `SCALE` | override total budget |
| `SEED` | 40 | mixing seed (**must match** across dump + reshuffle for reproducibility) |
| `COLLATE_DTYPES` | `instruction` | match production |
| `PACKED_IMAGES` | 24 | `max_images` cap |
| `SAMPLES_PER_SHARD` | 10000 | shard size |
| `DUMP_NUM_WORKERS` | 8 | dataloader workers/rank |
| `QOS` / `TIME` | from `.env` | queue overrides (e.g. debug) |

**Reshuffle** (`scripts/slurm/reshuffle/launcher.sh`):

| Var | Default | Meaning |
|---|---|---|
| `PHASE` | — (required) | `scatter` or `gather` |
| `OUT` | — (required) | finalized release root (holds `release/`) |
| `NODES` | 64 | → `WORLD_SIZE` (scatter markers target) |
| `NUM_BUCKETS` | 256 | intermediate buckets (gather markers target); use 64 at small scale |
| `SHUFFLE_BUFSIZE` | 20000 | gather streaming buffer (~4.6 GB RAM/rank; larger → Deff closer to 1) |
| `SAMPLES_PER_SHARD` | 10000 | output shard size |
| `SEED` | 40 | routing/shuffle seed |
| `RELEASE_DIR` | `$OUT/release` | override input dir |
| `ROOT` | `$OUT/reshuffle` | work root |

---

## Output layout

```
$OUT/
├── staging/                 # dump: per-rank shards (emptied by finalize)
├── markers/                 # dump: r<rank>.done
├── release/                 # STAGE-1 OUTPUT (biased)
│   ├── 000000.tar …
│   ├── dataset_info.json
│   └── meta_flat.json
└── reshuffle/
    ├── scatter/bucket_0000/ …   # transient (delete after finalize)
    ├── scatter_markers/         # r<rank>.done
    ├── shuffled/                # gather output (renamed by finalize)
    ├── gather_markers/          # b<bucket>.done
    └── release_shuffled/        # STAGE-2 OUTPUT (unbiased)
        ├── 000000.tar …
        ├── dataset_info.json    # globally_shuffled: true
        └── meta_flat.json
```

---

## Consuming the release

`meta_flat.json` is a single-entry, MAmmoTH-VL-shaped meta the trainer reads directly.
Each sample carries a `conversations.txt` (turns joined by `<EOCL>`, `<image>` inline) and
its images as `N.<ext>` in `<image>` order; on read-back `txt2conversations`
(`internvl/train/data/formatters/conversation.py`) reconstructs the turns. Text-only samples have no
image (a blank is regenerated at load). Provenance sidecars (`ds_name.txt`, `orig_url.txt`,
`data_type.txt`, `ntok.txt`, `mdp.txt`) are informational; the trainer uses `filters: {}`.

---

## Gotchas

- **Avoid reading shards on a login node** — a single 2–4 GB shard scan can hang for minutes;
  use the wrappers (`check_dump_release.sh`, `verify_source_dist.sh`, `assess_dispersion.sh`),
  which submit to `$INTERNET_PARTITION` via `srun`.
- **`SEED` must be identical** across dump and reshuffle if you ever re-run a pass — the
  streams are deterministic in `(seed, rank[, worker])`.
- **Node failures are expected at 128 nodes** — both dump and reshuffle are idempotent; just
  clean the partial tars for unfinished ranks/buckets and resubmit.
- **Decontam sidecars (`phash`/`sha256`/`sscd`) are intentionally omitted** — production
  `cleanup` drops them before the dump taps the stream, so training never saw them either.
