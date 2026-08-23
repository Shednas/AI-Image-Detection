# Early evaluation round, superseded

Twelve result files from the first evaluation round, kept as a record of that stage
of the work. **Nothing here may be used for any reported figure.** The metrics in the
live tree, `results/MODEL/stage_N/`, supersede all of it.

## Why this round was replaced

Six of these files score the models on an earlier evaluation set that was far easier
than the split the project settled on. CNN's accuracy on it is close to ceiling and
does not survive the move to the properly held-out split:

| Stage | Earlier set | | CNN accuracy | Held-out split | | CNN accuracy |
|---|---|---|---|---|---|---|
| | n | balance | | n | balance | |
| 1 | 76 | 38 / 38 | 97.37% | 82 | 42 / 40 | 79.27% |
| 2 | 902 | 451 / 451 | 99.78% | 839 | 424 / 415 | 83.55% |
| 3 | 3,228 | 1,614 / 1,614 | **99.44%** | 2,345 | 1,180 / 1,165 | **84.39%** |

A fifteen point drop at stage 3, and eighteen at stage 1, is the reason the earlier set
was abandoned. It is exactly balanced at every stage, which the held-out split is not,
and it evidently shared material with training. Any figure taken from it would
overstate every model.

## Direction of the historical FFT bias

**FFT was over-predicting real, not AI:** at stage 3 it labelled 2,151 of 2,345 images
real, catching only 194 of the 1,180 AI images, a 16.4 per cent AI recall, while
calling every one of the 1,165 real images real.

Recorded here because the opposite is easy to assume. The same collapse appears at
every stage: 82 of 82 predicted real at stage 1, 772 of 839 at stage 2. These are the
`holdout_split` files, which already used the held-out split rather than the earlier
set.

## These files are not label-inverted

Their ground-truth class assignment already matches the current
`{ai_generated: 0, real: 1}` convention. The three `fft_stage_N_holdout_split.json`
files use the same test sets as the live metrics, and their true-class row totals are
identical to those in `results/fft/stage_N/test_metrics.json`: 42 / 40 at stage 1,
424 / 415 at stage 2, 1,180 / 1,165 at stage 3.

An inverted mapping would have swapped those totals. It did not, so read these matrices
with the current convention and do not invert them a second time.

## What is here

| Original path | Archived as | Evaluated on |
|---|---|---|
| `results/cnn/stage_N/Biased_test.json` | `cnn_stage_N_earlier_set.json` | earlier set |
| `results/fft/stage_N/dataset_biased.json` | `fft_stage_N_earlier_set.json` | earlier set |
| `results/fft/stage_N/ai_test_biased.json` | `fft_stage_N_holdout_split.json` | held-out split |
| `results/fft/stage_N/training_ai_history.json` | `fft_stage_N_training_history.json` | training run |

The original names were misleading in two ways. "Biased" reads as a label defect when
it referred to the evaluation set, and `ai_test_biased` named the one group of files
that did **not** use that set.

The training histories carry `"model": "fft_improved"`, a label no current script
produces.

## Not read by anything

`report_metrics.py` loads three fixed filenames per model and stage,
`test_metrics.json`, `unseen/unseen_chameleon.json` and `unseen/unseen_mnw.json`. It
does not glob, so it never picked these up, and no other script in `research/` refers
to them by name. Moving and renaming them changed no output.
