# AI Image Detection

Undergraduate dissertation comparing four approaches to detecting AI-generated
images, and a web application that runs all four so their disagreements can be
inspected on a single image.

The four detectors are trained on the same data and evaluated on the same splits:

| Model | Approach |
|---|---|
| CNN | ResNet-50 backbone fine-tuned on spatial features |
| FFT | Learned weighting over concentric frequency bands |
| Hybrid | Late fusion of the CNN and FFT branches |
| STM | Handcrafted features (HOG, LBP, DCT, colour, noise) with LightGBM |

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
after Hybrid, while it detects only 35% of AI images in distribution, and 85% of
them from the unseen generators below. Read the two tables together without this
note and the numbers look contradictory.

The same models against MNW, 10,000 images from generators absent from the
training data. All are AI-generated, so the figure is the proportion correctly
flagged:

| Model | Detection rate |
|---|---|
| FFT | 85.20% |
| STM | 45.37% |
| CNN | 2.25% |
| Hybrid | 1.38% |

The ordering inverts. The two models that score highest in distribution detect
almost nothing from unseen generators, while the weakest in-distribution model
is the only one that generalises. This is the central finding of the
dissertation, and the reason the application exposes all four models rather than
picking the one with the best headline accuracy.

Every figure above is reproducible from the JSON in `research/results/`.

## Layout

```text
app/        Deployable web application. FastAPI backend, React frontend.
research/   Training and evaluation. Data pipeline, model definitions, results.
```

The two halves are independent. `app/` is submitted as a standalone deliverable
and does not import from `research/`. The four model definitions are duplicated
in both, kept in step by `app/backend/tests/test_model_sync.py`, which fails if
they drift apart structurally.

Start with [app/README.md](app/README.md) to run the application, or
[research/README.md](research/README.md) to retrain or reproduce the results.

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
