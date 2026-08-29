# Processed dataset

AI Image Detection, Sandesh Thapa, UON ID 24813026

Training data only. 15,497 images, already split and already preprocessed.

---

## Read this first

**You probably do not need this.** Every number in the dissertation is already
in the repository under `research/results/`, which is tracked in git. To print
them all:

```powershell
python scripts/report_metrics.py
```

No GPU, no dataset, no model weights required.

This zip is only for retraining the models yourself. A full run of all four
models across all three stages takes roughly a day on an RTX 3050.

**The labels contain noise.** Approximately 9% of images carry an incorrect
label, because CIFAKE was configured as a source for both classes. The
held-out test set is affected in the same proportion, so maximum attainable
accuracy against true labels is around 90.5%, not 100%. This is documented in
`research/README.md` and in the dissertation limitations. It applies equally to
every model, so comparisons between them are unaffected.

---

## Where to put it

Unzip so `processed/` lands directly inside `research/data/`:

```text
AI-Image-Detection/
`-- research/
    `-- data/
        `-- processed/
            |-- stage_1/
            |-- stage_2/
            `-- stage_3/
```

That folder already exists in the repository. Merge into it.

Check the paths from `research/`:

```powershell
python -c "from pathlib import Path; [print(s.name, sum(1 for p in s.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})) for s in sorted(Path('data/processed').glob('stage_*'))]"
```

Expected: `stage_1 500`, `stage_2 4999`, `stage_3 9998`.

Each folder holds only its own images. The stages are cumulative at load time,
so training at stage 2 uses stages 1 and 2 together and stage 3 uses all three,
giving 500, 5,499 and 15,497 images.

Each stage contains `train/`, `validation/` and `test/`, and each of those
contains `real/` and `ai_generated/`.

---

## Already done to these images

Converted to RGB, resized to 256x256 with LANCZOS, saved as JPEG at quality 95.
That is what `process_data.py` does and what the models were trained on.

**Do not run `process_data.py` on this data.** It rewrites in place, so a second
pass re-encodes already-encoded files and changes the pixels the models saw.

Some files end in `.png` but contain JPEG data, because the preprocessing step
re-encodes without renaming. Nothing breaks; PIL reads content, not extension.

---

## What you can run

Follow the Setup section of `research/README.md`, then skip Data steps 1 to 5
and unzip this instead. From `research/`:

```powershell
python -m src.training.train_cnn    --stage 3
python -m src.training.train_fft    --stage 3
python -m src.training.train_hybrid --stage 3
python -m src.training.train_stm    --stage 3
```

Variants kept for comparison:

```powershell
python -m src.training.train_fft_initial --stage 3
python -m src.training.train_hybrid_norm --stage 3
python -m src.training.train_hybrid_proj --stage 3
```

Held-out test set, plots and metrics. Both arguments required, `--model` is one
of `cnn`, `fft`, `hybrid`, `stm`:

```powershell
python -m src.evaluation.visualize --stage 3 --model cnn
```

`--stage` accepts 1, 2 or 3 everywhere and defaults to 2.

---

## What you cannot run

Anything involving the unseen generators, which means every `test_unseen_*`
script, the compression robustness results, and the matched-preprocessing
experiment that is the dissertation's central methodological finding.

Those all read from `research/data/unseen/`, which holds Chameleon and MNW and
is not part of this zip. That applies to both evaluation modes: the original
runs and the `--match_training_preprocessing` runs read from the same place, so
neither works without it.

The unseen sets are about 12GB. `research/README.md` explains how to obtain
them, and a mirror is linked in the top-level README.

## One convention that matters

Training used `{ai_generated: 0, real: 1}`. Real is the positive class, so a
model's sigmoid output is P(real), not P(AI), and every stored precision,
recall and F1 figure is measured on the real class. AI-class recall must be
derived from the confusion matrix as `cm[0][0] / sum(cm[0])`.

---

| | |
|---|---|
| Images | 15,497 across three cumulative stages |
| Split | 70 / 15 / 15, fixed seed 42 |
| Format | JPEG data, 256x256, quality 95, RGB |
| Size | approximately 430MB unzipped |
