# AI Image Detection, research

Training and evaluation for the four detectors. Three nested stages of
increasing size, so results can be read as a function of training data volume.

## Requirements

- Python 3.14.3
- A CUDA GPU. Training on CPU is impractical.

## Setup

```powershell
python -m venv dissertation_env
.\dissertation_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch first, from its own index. This is a separate step because an
`--index-url` inside `requirements.txt` would apply to every package in the
file, not just torch:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Then everything else:

```powershell
pip install -r requirements.txt
```

Confirm the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Built against torch 2.12.0+cu132. If your driver needs a different CUDA build,
change the index URL to match.

## Data

Nothing here is redistributed. `data/raw/` and `data/unseen/` ship as empty
directories with `.gitkeep` markers showing where each source belongs.

Scripted:

```powershell
python scripts/download_dataset.py forensynths --target 10000
python scripts/download_dataset.py --status-only   # report what is present
```

Manual, each needing an account or a manual export:

| Source | Destination | Notes |
|---|---|---|
| ImageNet | `data/raw/imagenet/` | registration required |
| Unsplash | `data/raw/unsplash/` | manual download, then `scripts/extract_unsplash.py` |
| COCO | `data/raw/coco/` | `train2017` and `val2017` |
| CIFAKE | `data/raw/cifake/` | `FAKE/` and `REAL/` |
| GenImage | `data/raw/genimage/` | one directory per generator |
| Flickr30k | `data/raw/flickr30k/` | |
| Chameleon | `data/unseen/chameleon/` | `fake/` and `real/` |

MNW is a git repository, so clone it rather than downloading by hand:

```powershell
git clone https://github.com/nsail-lab/MNW.git data/unseen/MNW
```

Once the sources are in place, build the splits and process them. The splitter
produces all three stages in one pass and guarantees no image is reused between
them:

```powershell
python scripts/split_dataset.py --seed 42
python scripts/process_data.py --stage 1
python scripts/process_data.py --stage 2
python scripts/process_data.py --stage 3
```

Output lands in `data/processed/stage_N/{train,validation,test}/{real,ai_generated}/`.
That directory is not tracked, since it is derived.

## Training

One command per model per stage:

```powershell
python -m src.training.train_cnn --stage 3
python -m src.training.train_fft --stage 3
python -m src.training.train_hybrid --stage 3
python -m src.training.train_stm --stage 3
```

`--stage` accepts 1, 2 or 3 and defaults to 2. `train_fft_initial` trains the
earlier FFT variant kept for comparison.

Checkpoints go to `checkpoints/<model>/stage_N/`, metrics and plots to
`results/<model>/stage_N/`. Checkpoints are not tracked; results are.

## Evaluation

Plots and test metrics for a trained checkpoint:

```powershell
python -m src.evaluation.visualize --stage 3 --model cnn
```

Unseen generators, with optional JPEG degradation:

```powershell
python -m src.evaluation.test_unseen_cnn --stage 3 --dataset all --degradation all
```

`--dataset` takes `chameleon`, `mnw` or `all`. `--degradation` takes `none`,
`light`, `heavy` or `all`. The other three models have matching modules, and
`test_unseen_fft` additionally takes `--model initial|improved`.

Both commands are a straight cross-product over model and stage, so the full
matrix is twelve invocations of each.

## Results

`results/<model>/stage_N/` holds `test_metrics.json`, `training_history.json`,
`plots/`, and for stage 3 an `unseen/` directory with the Chameleon and MNW
evaluations. Headline numbers are in the root [README](../README.md).

Labels follow `{ai_generated: 0, real: 1}` throughout, so a model's sigmoid
output is P(real), not P(AI). Every metric in this tree depends on that mapping.
Reversing it inverts AUC and turns a detection rate into a miss rate.

## Layout

```text
data/           raw sources, processed splits, metadata
scripts/        download, split and preprocess
src/models/     the four model definitions, duplicated in app/backend/models/
src/training/   per-model entry points and the shared trainer
src/evaluation/ metrics, visualisation, unseen-generator tests
results/        metrics and figures, tracked
checkpoints/    trained weights, not tracked
```
