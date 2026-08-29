# AI Image Detection

Undergraduate dissertation comparing four approaches to detecting AI-generated
images, and a web application that runs all four so their disagreements can be
inspected on a single image.

Start with [app/README.md](app/README.md) to run the application, or
[research/README.md](research/README.md) to retrain or reproduce the results.

The four detectors are trained on the same data and evaluated on the same splits:

| Model | Approach |
|---|---|
| CNN | ResNet-50 backbone fine-tuned on spatial features |
| FFT | Magnitude spectrum split into four concentric bands, each convolved and pooled |
| Hybrid | Late fusion of the CNN and FFT branches |
| STM | Handcrafted features (HOG, LBP, DCT, colour, noise) with LightGBM |

FFT carries a learnable weight per band, but it did not train away from its
uniform initialisation. The Stage 3 weights are 0.2583, 0.2473, 0.2381 and
0.2562, a maximum deviation of 0.054, so in practice the four bands are averaged
rather than weighted.

## Results

Stage 3, held-out test set drawn from the training distribution:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| CNN | 84.39% | 79.27% | 92.88% | 85.53% | 93.83% |
| Hybrid | 84.18% | 77.46% | 96.14% | 85.79% | 93.94% |
| STM | 75.35% | 74.28% | 77.08% | 75.65% | 83.59% |
| FFT | 61.45% | 57.28% | 88.15% | 69.44% | 67.05% |

Training used `{ai_generated: 0, real: 1}`, so real is the positive class and
the precision and recall above are **for the real class**. Recall of 96.14%
means Hybrid correctly identifies 96% of genuine photographs, not that it
catches 96% of AI images. Derived from the same confusion matrices, recall on
the AI class is:

| Model | Real recall (above) | AI recall |
|---|---|---|
| CNN | 92.88% | 76.02% |
| Hybrid | 96.14% | 72.37% |
| STM | 77.08% | 73.64% |
| FFT | 88.15% | 35.08% |

FFT is the reason this matters. Its 88.15% in the table is the highest recall
after Hybrid, while it detects only 35% of AI images in distribution. Read the
two tables together without this note and the numbers look contradictory.

The same models against MNW, 10,000 images from generators absent from the
training data. All are AI-generated, so the figure is the proportion correctly
flagged:

| Model | Mismatched preprocessing | Matched preprocessing |
|---|---|---|
| FFT | 85.20% | 38.24% |
| STM | 45.37% | 27.61% |
| CNN | 2.25% | 10.98% |
| Hybrid | 1.38% | 10.67% |

The first column evaluates the unseen images as they arrive. The second puts
them through the same steps the training data went through: RGB, 256x256 LANCZOS,
JPEG quality 95. The models were trained on images rewritten that way, so the
first column measures a preprocessing mismatch as well as a generalisation gap.

The ordering inverts in both columns, so the finding holds: the models that score
highest in distribution detect least from unseen generators, and the weakest
in-distribution model generalises best. Its size does not hold. Matched, FFT
leads CNN by about three and a half times rather than thirty-eight, so most of
the original gap was measurement error rather than a property of the models.

That correction is itself a finding, and it is the reason the application exposes
all four models rather than picking the one with the best headline accuracy.

Roughly 9% of the dataset carries an incorrect label, because CIFAKE was
configured as a source for both classes. The held-out test set is affected in the
same proportion, so the attainable ceiling is about 90.5% rather than 100%. It
applies equally to every model, so comparisons between them stand. See
[research/README.md](research/README.md) for how to measure it.

The underlying JSON for every figure above is in `research/results/`.
`research/scripts/report_metrics.py` tabulates the held-out and unseen numbers;
the matched-preprocessing column is in `research/results/_preproc/`, which that
script does not read.

## Layout

```text
app/        Deployable web application. FastAPI backend, React frontend.
research/   Training and evaluation. Data pipeline, model definitions, results.
```

The two halves are independent. `app/` is submitted as a standalone deliverable
and does not import from `research/`. The four model definitions are duplicated
in both, kept in step by `app/backend/tests/test_model_sync.py`, which fails if
they drift apart structurally.

## Model weights

Four files, all Stage 3 checkpoints:

| File | Size |
|---|---|
| `best_cnn.pt` | 90MB |
| `best_hybrid.pt` | 96MB |
| `best_fft.pt` | 1.7MB |
| `stm_model.joblib` | 0.9MB |

They belong in `app/backend/models/weights/`.

**The submitted zip already contains them, so there is nothing to download.**
They are excluded from git rather than from the submission: `best_cnn.pt` and
`best_hybrid.pt` are past GitHub's 50MB warning and near its 100MB hard limit,
so a clone of this repository will find that directory holding only a
`.gitkeep`. Using checkpoints from a different stage will produce numbers that
disagree with the dissertation.

## Licence

MIT, see [LICENSE](LICENSE). This covers the code only. The datasets are
redistributed by nobody here and must be obtained from their original sources
under their own licences.
