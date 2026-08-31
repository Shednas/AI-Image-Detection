# AI Image Detection, research

Training and evaluation for the four detectors. Three cumulative stages of
increasing size, so results can be read as a function of training data volume.

---

## Read this first

**Nothing here needs to be run to verify a claim in the dissertation.** Every
figure is already in `research/results/`, which is tracked in git. To print them
all:

```powershell
cd research
python scripts/report_metrics.py
```

No GPU, no dataset and no model weights are needed for that. The
matched-preprocessing runs live in `results/_preproc/` and are read directly as
JSON, since `report_metrics.py` does not cover them.

Everything below is for reproducing those results from scratch.

---

## Which route do you need?

Pick one. Each row lists what you must download and roughly how long it takes.

| Goal | You need | Time |
|---|---|---|
| Read the results | Nothing. They are in `results/` | Minutes |
| Re-run the held-out evaluation and plots | `checkpoints/` and `data/processed/` | 1 to 2 hours |
| Re-run the unseen-generator evaluation | `checkpoints/` and `data/unseen/` | 6 to 8 hours |
| Retrain the models | `data/processed/` | 6 to 8 hours |
| Rebuild the dataset from source | `data/raw/` | 2 to 4 hours, plus download |

Times assume an RTX 3050 with 4GB of VRAM and will vary with hardware and
network speed.

**You do not need to retrain in order to evaluate.** The trained checkpoints are
provided.

---

## Downloads

Everything that is not in the repository is in one Google Drive folder:

**[Google Drive folder](https://drive.google.com/drive/folders/1Ozr4LUUvmH9a7LNGHMbamnAF8oJRPCK4)**

The same link is in the NILE submission. About 68GB in total, so allow several
hours. Read the README at the root of that folder before downloading anything.

| Folder | Contents | Unzip to |
|---|---|---|
| `checkpoints/` | Trained weights for every model at every stage | `research/checkpoints/` |
| `data/raw/` | The seven source datasets, unprocessed | `research/data/raw/` |
| `data/processed/` | The training set, already split and preprocessed | `research/data/processed/` |
| `data/unseen/` | Chameleon and MNW, for unseen-generator evaluation | `research/data/unseen/` |

`data/processed/` is also submitted separately to NILE as
`processed_dataset.zip`, which is 430MB rather than the full download.

**If you only want to re-run evaluations**, take `checkpoints/` and
`data/unseen/`. You can skip `data/raw/` entirely.

---

## Setup

Do this regardless of which route you are taking.

**Requirements:** Python 3.14.3, and a CUDA GPU. Training on CPU is
impractical.

### Step 1: Create the environment

```powershell
cd research
python -m venv dissertation_env
.\dissertation_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch

Separately from everything else, because an `--index-url` inside
`requirements.txt` applies to every package in the file rather than just torch.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Built against torch 2.12.0+cu132. Change the index URL if your driver needs a
different CUDA build.

### Step 3: Install everything else

```powershell
pip install -r requirements.txt
```

### Step 4: Confirm the GPU is visible

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**Expected:** `True` followed by a CUDA version. If it prints `False`, training
falls back to CPU and becomes impractical.

**Every command below runs from `research/` with the environment activated.** If
you open a new terminal, or the prompt does not begin with `(dissertation_env)`:

```powershell
cd research
.\dissertation_env\Scripts\Activate.ps1
```

From the repository root, `cd research` first. From `app/`, `cd ..\research`.

---

## Check before you run anything

Each section below opens with a check. Run it and confirm the expected output
before continuing. All four are read-only and take seconds.

### Check A: the raw dataset is complete

Only needed if you are rebuilding the dataset from source.

```powershell
python scripts/download_dataset.py --status-only
```

**Expected:** all seven sources report OK, with counts close to these.

| Source | Files |
|---|---|
| ImageNet | 50,001 |
| COCO | 123,289 |
| Unsplash | 4,003 |
| GenImage | 122,305 |
| ForenSynths | 10,001 |
| CIFAKE | 109,860 |
| Flickr30k | 31,785 |

Anything reading NOT FOUND or EMPTY means that source is missing or in the
wrong folder. Fix it before running `split_dataset.py`.

**This check looks at `data/raw/` only.** It never inspects `data/unseen/`, so
passing it does not mean the unseen evaluation will work. Use Check C for that.

### Check B: the processed dataset is in place

Needed for training, and for the held-out evaluation.

```powershell
python -c "from pathlib import Path; [print(s.name, sum(1 for p in s.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})) for s in sorted(Path('data/processed').glob('stage_*'))]"
```

**Expected:** `stage_1 500`, `stage_2 4999`, `stage_3 9998`.

Each folder holds only its own images. The stages are cumulative at load time,
so training at stage 3 uses all three folders together, giving 15,497 images.

If the command prints nothing at all, `data/processed/` holds no `stage_*`
folders and the unzip landed in the wrong place. `stage_1` should sit directly
inside `research/data/processed/`. A stage that is listed but reports zero is
present and empty.

### Check C: the unseen datasets are in place

Needed for any `test_unseen_*` script.

```powershell
python -c "from pathlib import Path; [print(p, sum(1 for f in Path(p).rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})) for p in ['data/unseen/chameleon/real','data/unseen/chameleon/fake','data/unseen/MNW']]"
```

**Expected:** roughly `chameleon/real 14863`, `chameleon/fake 11170`,
`MNW 11290`.

MNW is a git repository rather than a plain folder. If the count is zero after
cloning, it uses Git LFS and the files are pointers rather than images. Run
`git lfs install`, then `git lfs pull` inside that folder.

### Check D: the checkpoints are in place

Needed for any evaluation.

```powershell
python -c "from pathlib import Path; [print(p.parent.parent.name, p.name) for p in sorted(Path('checkpoints').rglob('*')) if p.suffix in {'.pt','.joblib'}]"
```

**Expected:** at minimum, one file under each of `checkpoints/cnn/stage_3/`,
`checkpoints/fft/stage_3/`, `checkpoints/hybrid/stage_3/` and
`checkpoints/stm/stage_3/`.

An empty list means the download did not unzip into the right place, or you are
not in `research/`.

**A shortcut if you only need Stage 3.** The four Stage 3 weights also ship with
the application. Copy them across rather than downloading `checkpoints/`:

| From `app/backend/models/weights/` | To |
|---|---|
| `best_cnn.pt` | `research/checkpoints/cnn/stage_3/` |
| `best_fft.pt` | `research/checkpoints/fft/stage_3/` |
| `best_hybrid.pt` | `research/checkpoints/hybrid/stage_3/` |
| `stm_model.joblib` | `research/checkpoints/stm/stage_3/` |

Stages 1 and 2 and the variant architectures are only in the Drive folder.

---

## Dataset preparation

**Skip this section if you downloaded `data/processed/`.** It is already done.

Run Check A first.

### Step 1: Build the splits

```powershell
python scripts/split_dataset.py --seed 42
```

Produces all three stages in one pass and guarantees no image is reused between
them. `--seed` is the only argument; there is no `--stage`.

### Step 2: Validate and resize

```powershell
python scripts/process_data.py --stage 1
python scripts/process_data.py --stage 2
python scripts/process_data.py --stage 3
```

Also takes `--target-size`, default 256, and `--dry-run` to report without
writing or deleting.

**This step is what the models were actually trained on.** Each image is
converted to RGB, resized to 256x256 with LANCZOS, and re-saved as JPEG quality
95, in place. That matters for evaluation, see `--match_training_preprocessing`
below.

**Do not run this on a downloaded `data/processed/`.** It rewrites in place, so
a second pass re-encodes already-encoded files and changes the pixels the models
were trained on.

Output lands in
`data/processed/stage_N/{train,validation,test}/{real,ai_generated}/`. That
directory is derived and is not tracked.

Run Check B afterwards.

---

## Training

**Skip this section if you downloaded `checkpoints/`.** The trained weights are
provided.

Run Check B first.

Every training script takes `--stage`, accepting 1, 2 or 3. All default to 2
except `train_fft_initial`, which defaults to 1. All except `train_stm` and
`train_fft_initial` also take `--epochs` to override the per-model default.

```powershell
python -m src.training.train_cnn --stage 3
python -m src.training.train_fft --stage 3
python -m src.training.train_hybrid --stage 3
python -m src.training.train_stm --stage 3
```

All four models at stage 3 takes roughly two hours. All three stages for all
four models takes roughly a day.

### Variants kept for comparison

```powershell
python -m src.training.train_fft_initial --stage 3
python -m src.training.train_hybrid_norm --stage 3
python -m src.training.train_hybrid_proj --stage 3
```

`train_fft_initial` trains the earlier FFT architecture, `FFTDetectorInitial`.
It takes `--stage` only.

`train_hybrid_norm` trains `HybridNormDetector`, which L2 normalises both
branches before concatenation. The branches are 2048 and 256 dimensions with
different natural scales, so unnormalised concatenation lets the CNN side
dominate the fusion input.

`train_hybrid_proj` trains `HybridProjDetector`, which normalises both branches
and then projects the CNN branch from 2048 down to 256, so the two enter fusion
at equal width as well as equal scale. Fusion input is 512 rather than 2304.

Each writes to its own checkpoint and results tree with its own filename, so no
variant can overwrite another:

| Script | Checkpoint | Results |
|---|---|---|
| `train_hybrid` | `checkpoints/hybrid/stage_N/best_hybrid.pt` | `results/hybrid/stage_N/` |
| `train_hybrid_norm` | `checkpoints/hybrid_norm/stage_N/best_hybrid_norm.pt` | `results/hybrid_norm/stage_N/` |
| `train_hybrid_proj` | `checkpoints/hybrid_proj/stage_N/best_hybrid_proj.pt` | `results/hybrid_proj/stage_N/` |

Checkpoints are not tracked in git; results are.

---

## Evaluation

Run Check D first. For the unseen tests, run Check C as well.

### Held-out test set, plots and metrics

Needs `checkpoints/` and `data/processed/`.

```powershell
python -m src.evaluation.visualize --stage 3 --model cnn
```

Both arguments are required. `--model` takes `cnn`, `fft`, `hybrid` or `stm`.
There is no option for the variant architectures.

### Unseen generators

Needs `checkpoints/` and `data/unseen/`. Roughly 6 to 8 hours for all four
models across both datasets and all three degradation levels.

```powershell
python -m src.evaluation.test_unseen_cnn --stage 3 --dataset all --degradation all --n_samples 10000
python -m src.evaluation.test_unseen_fft --stage 3 --dataset all --degradation all --n_samples 10000
python -m src.evaluation.test_unseen_hybrid --stage 3 --dataset all --degradation all --n_samples 10000
python -m src.evaluation.test_unseen_stm --stage 3 --dataset all --degradation all --n_samples 10000
```

All four share these arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--stage` | 3 | 1, 2 or 3 |
| `--dataset` | `chameleon` | `chameleon`, `mnw` or `all` |
| `--degradation` | `none` | `none`, `light`, `heavy` or `all`. JPEG re-encode at quality 75 and 25 |
| `--n_samples` | 200 | images per class, see the warning below |
| `--results_dir` | derived from `--stage` | where to write, so a comparison run cannot overwrite a published result |
| `--match_training_preprocessing` | off | see below |
| `--dump_probabilities` | off | see below |

`test_unseen_hybrid` additionally takes `--checkpoint` to load a specific file,
and `--model_class` accepting `hybrid`, `hybrid_norm` or `hybrid_proj`, which
selects the architecture to build before loading. The other three load a fixed
path under `checkpoints/MODEL/stage_N/`, and STM globs for `*.joblib` there
since `train_stm` writes a run-specific filename.

**`test_unseen_fft` does not take a `--model` argument.** It always loads
`checkpoints/fft/stage_N/best_fft.pt`. Nothing evaluates the
`train_fft_initial` checkpoint through this path.

### `--n_samples` is per class and it matters

It defaults to **200**. Every published result uses **10000**, which gives
10,000 real and 10,000 AI on Chameleon, and 10,000 AI on MNW. Omitting it
produces a different, much smaller sample that is not comparable with anything
in the dissertation. The seed is fixed at 42, so passing the same count
reproduces the same images.

### `--match_training_preprocessing`

Off by default, which preserves the behaviour every published result was
produced with.

The models were trained on images that `process_data.py` had already rewritten
on disk: RGB, 256x256 via LANCZOS, JPEG quality 95. The unseen sets were never
put through that step, so by default they reach the models at their original
size, resized with bilinear interpolation and with no JPEG re-encode. This flag
reproduces the training pipeline in memory, before the standard transform and
before any degradation step, so an unseen set can be scored through the same
chain the models were trained on. `data/unseen/` is never written to.

**Results are not comparable across the two settings.** A run with the flag on
and a run with it off are measuring different inputs, so always record which was
used. The output JSON carries a `match_training_preprocessing` field for exactly
this reason. Use `--results_dir` to keep the two apart:

```powershell
python -m src.evaluation.test_unseen_fft --stage 3 --dataset mnw --degradation none --n_samples 10000 --match_training_preprocessing --results_dir results/_preproc/fft_mnw_on
```

### `--dump_probabilities`

Off by default. When on, writes `unseen_DATASET_probabilities.json` beside the
metrics, holding one row per image with its filename, true label and raw
probability. A threshold sweep then becomes a laptop job rather than another GPU
run.

The probability field is named `p_real`, not `prob`. Training used
`{ai_generated: 0, real: 1}`, so the raw sigmoid is P(real) and
**P(AI) = 1 - p_real**. `label` uses the same mapping. The file repeats this in
a `note` field.

### Tabulate every recorded metric

```powershell
python scripts/report_metrics.py
python scripts/report_metrics.py --csv metrics.csv
```

Prints every metric per model per stage to four decimal places, so the
dissertation can be checked against the files mechanically rather than by eye.
Runs from any working directory.

---

## Results

`results/MODEL/stage_N/` holds `test_metrics.json`, `training_history.json`,
`plots/`, and for stage 3 an `unseen/` directory with the Chameleon and MNW
evaluations. Headline numbers are in the root [README](../README.md).

`plots/` exists for the four primary models across all three stages. The variant
trees and the `_preproc/` runs record metrics only, since `visualize.py` covers
the four primary models.

### Reading the stored metrics

**Labels follow `{ai_generated: 0, real: 1}` throughout**, so a model's sigmoid
output is P(real), not P(AI). Every metric in this tree depends on that mapping.
Reversing it inverts AUC and turns a detection rate into a miss rate.

Because real is the positive class, the stored `precision`, `recall` and `f1`
are all **for the real class**. Recall of 96% means the model identifies 96% of
genuine photographs, not that it catches 96% of AI images. AI-class recall is
not stored and must be derived from the confusion matrix as
`cm[0][0] / sum(cm[0])`.

MNW contains no real images, so ROC AUC is undefined there and precision, recall
and F1 are zero by construction. Only the detection rate carries information,
and it appears as `accuracy` in every script except `test_unseen_cnn`, which
reports it as `detection_rate`.

---

## Label contamination

**Roughly 9% of the dataset carries an incorrect label.**

`split_dataset.py` lists `cifake` under both the real and the fake source maps,
and `get_images_from_source` recurses, so both labels drew from the same pool:
the 109,858 images under `data/raw/cifake/FAKE/` and `data/raw/cifake/REAL/`
together. Neither subfolder is filtered by label.

The held-out test set is affected in the same proportion, so attainable accuracy
against true labels is about 90.5% rather than 100%. It applies equally to every
model, so comparisons between them are unaffected, but no single model's
accuracy should be read as its ceiling.

The unseen datasets never pass through this splitter, so all
out-of-distribution results are unaffected.

To measure it on the current splits:

```powershell
python check_cifake_labels.py
```

Read-only. It matches each processed file back to the raw folder it came from
and reports those that landed under the opposite label, counting names present
in both raw folders separately as untraceable rather than assuming they are
wrong.

---

## Layout

```text
data/           raw sources, processed splits, metadata
scripts/        download, split, preprocess, report
src/models/     the four model definitions, duplicated in app/backend/models/
src/training/   per-model entry points and the shared trainer
src/evaluation/ metrics, visualisation, unseen-generator tests
results/        metrics and figures, tracked
checkpoints/    trained weights, not tracked
```
