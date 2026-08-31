# AI Image Detection, research

Training and evaluation for the four detectors. Three cumulative stages of
increasing size, so results can be read as a function of training data volume.

## Read this first

**Nothing here needs retraining to verify a claim.** Every number in the
dissertation is already in `research/results/`, which is tracked in git. To print
them all:

```powershell
python scripts/report_metrics.py
```

No GPU, no dataset and no model weights are required for that. The
matched-preprocessing runs are in `results/_preproc/`, which
`report_metrics.py` does not read; those JSON files are read directly.

The four Stage 3 weights also ship with the application, in
`app/backend/models/weights/`. Copy them into
`research/checkpoints/MODEL/stage_3/` to run `visualize.py` or the unseen tests
without retraining:

| From `app/backend/models/weights/` | To |
|---|---|
| `best_cnn.pt` | `research/checkpoints/cnn/stage_3/` |
| `best_fft.pt` | `research/checkpoints/fft/stage_3/` |
| `best_hybrid.pt` | `research/checkpoints/hybrid/stage_3/` |
| `stm_model.joblib` | `research/checkpoints/stm/stage_3/` |

The unseen tests additionally need `data/unseen/`, which is not in the
repository. See the next section.

### Getting the data

The datasets and the trained checkpoints for every stage are in a Google Drive
folder:

**[Google Drive folder](https://drive.google.com/drive/folders/1Ozr4LUUvmH9a7LNGHMbamnAF8oJRPCK4)**

The same link is in the NILE submission. About 68GB in total, so allow several
hours. The README at the root of that folder explains what each part is for and
where it goes.

If you only want to retrain the models, the processed training set alone is
430MB and is submitted to NILE as `processed_dataset.zip`. Unzip it into
`data/processed/` and skip the whole Data section below.

If you only want to re-run evaluations, take `checkpoints/` from the Drive
folder, or copy the four Stage 3 weights from the application as shown above.

## Label contamination

**Roughly 9% of the dataset carries an incorrect label.** `split_dataset.py`
lists `cifake` under both the real and the fake source maps, and
`get_images_from_source` recurses, so both labels drew from the same pool: the
109,858 images under `data/raw/cifake/FAKE/` and `data/raw/cifake/REAL/`
together. Neither subfolder is filtered by label.

The held-out test set is affected in the same proportion, so the attainable
accuracy against true labels is about 90.5%, not 100%. It applies equally to
every model, so comparisons between them are unaffected, but no single model's
accuracy should be read as its ceiling.

To measure it on the current splits:

```powershell
python check_cifake_labels.py
```

Read-only. It matches each copied file back to the raw folder it came from and
reports the ones that landed under the opposite label, counting names that appear
in both raw folders separately as untraceable rather than assuming they are wrong.

## Requirements

- Python 3.14.3
- A CUDA GPU. Training on CPU is impractical.

---

## Setup

### Step 1: Create the environment

```powershell
cd research
python -m venv dissertation_env
.\dissertation_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch

Separately, because an `--index-url` inside `requirements.txt` applies to every
package in the file rather than just torch.

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

Every command below runs from `research/` with the environment activated.

---

## Data

The datasets are not tracked in git. `data/raw/`, `data/processed/`,
`data/unseen/` and `checkpoints/` ship as empty directories with `.gitkeep`
markers showing where each source belongs.

Follow this section only if you are rebuilding the dataset from its original
sources. To use the supplied copies instead, see Getting the data above.

### Step 1: Download the sources

One source is scripted. `download_dataset.py` takes a dataset name, and
`forensynths` is the only accepted value:

```powershell
python scripts/download_dataset.py forensynths --target 10000
```

It also takes `--output-dir` to override the destination, and `--status-only` to
report what is present without downloading.

The rest are manual, each needing an account or a manual export:

| Source | Destination | Notes |
|---|---|---|
| ImageNet | `data/raw/imagenet/` | registration required |
| COCO | `data/raw/coco/` | `train2017` and `val2017` |
| CIFAKE | `data/raw/cifake/` | `FAKE/` and `REAL/` |
| GenImage | `data/raw/genimage/` | one directory per generator |
| Flickr30k | `data/raw/flickr30k/` | |
| Unsplash | `data/raw/unsplash/` | save the manifest as `photos.csv000` here, then step 2 |
| Chameleon | `data/unseen/chameleon/` | `fake/` and `real/` |

MNW is a git repository, so clone it:

```powershell
git clone https://github.com/nsail-lab/MNW.git data/unseen/MNW
```

### Step 2: Fetch the Unsplash images

```powershell
python scripts/extract_unsplash.py --target 4000
```

The manifest must be named `photos.csv000` and sit in `data/raw/unsplash/`.
The Unsplash Lite archive ships it tab-separated despite the extension, which is
what the script expects. Use `--metadata-file` to point elsewhere.

Resumable: already-downloaded URLs are tracked in
`data/raw/unsplash/downloaded_urls.txt` and skipped. Also takes `--output-dir`,
`--metadata-file`, `--random-sample`, `--delay`, and `--verify` to count what is
already present without downloading.

### Step 3: Check what is present

```powershell
python scripts/download_dataset.py --status-only
```

All seven sources should read OK before continuing.

Two limits of this check. It looks at `data/raw/` only and never at
`data/unseen/`, so Chameleon and MNW can be missing while it reports everything
fine, and the failure only surfaces during evaluation. It also reports OK for
any count above zero, so a partially completed download still passes. Confirm
the unseen sets by hand.

### Step 4: Build the splits

```powershell
python scripts/split_dataset.py --seed 42
```

Produces all three stages in one pass and guarantees no image is reused between
them. `--seed` is the only argument; there is no `--stage`.

### Step 5: Validate and resize

```powershell
python scripts/process_data.py --stage 1
python scripts/process_data.py --stage 2
python scripts/process_data.py --stage 3
```

Also takes `--target-size` (default 256) and `--dry-run` to report without
writing or deleting.

**This step is what the models were actually trained on.** Each image is
converted to RGB, resized to 256x256 with LANCZOS, and re-saved as JPEG quality
95, in place. It matters for evaluation, see the preprocessing note below.

Output lands in
`data/processed/stage_N/{train,validation,test}/{real,ai_generated}/`. That
directory is derived and is not tracked.

---

## Training

Every training script takes `--stage`, which accepts 1, 2 or 3 and defaults to 2.
All except `train_stm` and `train_fft_initial` also take `--epochs` to override
the per-model default.

```powershell
python -m src.training.train_cnn --stage 3
python -m src.training.train_fft --stage 3
python -m src.training.train_hybrid --stage 3
python -m src.training.train_stm --stage 3
```

### Variants kept for comparison

```powershell
python -m src.training.train_fft_initial --stage 3
python -m src.training.train_hybrid_norm --stage 3
python -m src.training.train_hybrid_proj --stage 3
```

`train_fft_initial` trains the earlier FFT architecture, `FFTDetectorInitial`.
It takes `--stage` only.

`train_hybrid_norm` trains `HybridNormDetector`, which L2 normalises both
branches before concatenation. The two branches are 2048 and 256 dimensions with
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

Checkpoints are not tracked; results are.

---

## Evaluation

### Held-out test set, plots and metrics

```powershell
python -m src.evaluation.visualize --stage 3 --model cnn
```

Both arguments are required. `--model` takes `cnn`, `fft`, `hybrid` or `stm`,
and there is no option for the variant architectures.

### Unseen generators

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
| `--degradation` | `none` | `none`, `light`, `heavy` or `all`, JPEG re-encode at quality 75 and 25 |
| `--n_samples` | 200 | images per class, see the warning below |
| `--results_dir` | derived from `--stage` | where to write, so a comparison run cannot overwrite a published result |
| `--match_training_preprocessing` | off | see below |
| `--dump_probabilities` | off | see below |

`test_unseen_hybrid` additionally takes `--checkpoint` to load a specific file,
and `--model_class` accepting `hybrid`, `hybrid_norm` or `hybrid_proj`, which
selects the architecture to build before loading. The other three load a fixed
checkpoint: `checkpoints/MODEL/stage_N/`, and STM globs for `*.joblib` there
since `train_stm` writes a run-specific filename.

**`test_unseen_fft` does not take a `--model` argument.** It always loads
`checkpoints/fft/stage_N/best_fft.pt`. Nothing evaluates the `train_fft_initial`
checkpoint through this path.

### `--n_samples` is per class and it matters

It defaults to **200**. Every published result uses **10000**, which gives 10,000
real and 10,000 AI on Chameleon, and 10,000 AI on MNW. Omitting it produces a
different, much smaller sample that is not comparable with anything in the
dissertation. The seed is fixed at 42, so passing the same count reproduces the
same images.

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

`plots/` exists for the four primary models across all three stages, while the
variant trees and the `_preproc/` runs record metrics only, since `visualize.py`
covers the four primary models.

**Labels follow `{ai_generated: 0, real: 1}` throughout**, so a model's sigmoid
output is P(real), not P(AI). Every metric in this tree depends on that mapping.
Reversing it inverts AUC and turns a detection rate into a miss rate.

Because real is the positive class, the stored `precision`, `recall` and `f1`
are all **for the real class**. Recall of 96% means the model identifies 96% of
genuine photographs, not that it catches 96% of AI images. AI-class recall is
not stored and must be derived from the confusion matrix as
`cm[0][0] / sum(cm[0])`.

MNW contains no real images, so ROC AUC is undefined there and precision, recall
and F1 are zero by construction. Only the detection rate carries information, and
it appears as `accuracy` in every script except `test_unseen_cnn`, which reports
it as `detection_rate`.

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
