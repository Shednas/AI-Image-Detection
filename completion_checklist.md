# Dissertation Completion Checklist

**Submission: 5 September 2026**
Final Report · Source Code (Word) · Source Code (Executable) · Recorded Viva and Demonstration

Interim mark: 70/100 (B+)

Tick items as you go. Phases are ordered by dependency, not by importance.

---

## Phase 0: Repository setup

Half a day. Do this before touching code so every fix lands in version control.

- [ ] Create repo `ai-image-detection`, public
- [ ] Restructure into `research/` and `app/`
- [ ] Point `app/backend/pipeline.py` imports at `research/src/models/`
- [ ] Add `.gitignore` at root (data, checkpoints, venv, node_modules)
- [ ] Add root `README.md`, `research/README.md`, `app/README.md`
- [ ] Add `requirements.txt` to `research/` and `app/backend/`
- [ ] Add `LICENSE` (MIT)
- [ ] Set repo description and topics
- [ ] Upload the four weight files to a GitHub Release, link from `app/README.md`
- [ ] Verify: fresh clone plus release download plus README steps gets the app running
- [ ] After the first push to GitHub, relocate the repo outside the OneDrive-synced
      tree. OneDrive can corrupt a `.git` directory by syncing it mid-write, and
      until that first push the local copy is the only copy of the history. GitHub
      becomes the backup at that point, which is why this is safe to do after the
      push and not before.

---

## Phase 1: Bug fixes

Two days. These are live defects, not improvements. Two of them would visibly
contradict themselves during the recorded demonstration.

### 1.1 P(AI) propagation

`probability` currently means P(real) everywhere, but the Batch page labels that
column "P(AI)". One image gives contradictory numbers across pages.

Audited against the code on 27 July 2026. The pipeline, `results.py` and the
analyze endpoint are already correct. The batch endpoint is the only live defect,
and it is the worst variant: analyze writes P(AI) into `consensus_score` while
batch writes P(real) into the same column, with nothing to tell the two apart.

- [x] `pipeline.py`: add `p_ai` to the `predict` return dict
- [x] `results.py`: read `raw_output["p_ai"]` instead of recomputing `1.0 - prob`
- [x] `main.py`: store `raw["p_ai"]` in `consensus_score` (analyze endpoint)
- [x] `main.py`: store `raw["p_ai"]` in `predicted_probability` (analyze endpoint)
- [x] `main.py`: same two fields in the batch endpoint loop
- [x] `BatchPage.jsx`: render `row.p_ai`
- [x] Rename `probability` to `p_real` throughout (pipeline, results, batch rows).
      Do not drop it: `format_single` still needs it for `confidence_pct`.
      Renaming makes any stale `row.probability` reference fail visibly rather
      than silently invert.
- [x] `HistoryPage.jsx`: add a P(AI) column. `db.get_history` returned the value
      as `score` and the frontend discarded it, so History displayed no
      probability at all. The key is now `p_ai`, matching the rest of the codebase.
- [x] ~~Truncate all four tables~~ Not required: the tables were empty when the
      fixes landed, so no row ever held the old meaning. Rows written since are
      all P(AI). The column type also changed, so a stale database elsewhere needs
      DROP rather than TRUNCATE. See 1.6.
- [ ] **Verify:** same AI image through Analyze, Batch, and History shows one identical percentage

### 1.3 Grad-CAM direction

`output.mean().backward()` always computes gradients toward higher logit, which
means toward "real". The heatmap explains the wrong verdict half the time.

- [x] Pass the verdict into `_generate_gradcam`
- [x] Negate the backward target when verdict is `AI_GENERATED`
- [x] Update the caption text if the wording no longer fits
- [x] Caption fix for Hybrid: the heatmap hooks `cnn_branch.layer4`, so it
      reflects the spatial branch specifically, not the fused decision. Say so.
      The spectrogram covers the frequency branch, so both are represented.
- [x] **Verify:** same image and same CNN weights through both verdict paths
      produce different heatmaps, 93.7% of pixels differing by more than 8/255.
      Checked at unit level on 27 July 2026. The two-image version of this check
      belongs in the Phase 2.7 pass, where the screenshots get taken anyway.

### 1.4 Session reuse

Every request calls `create_session()`, so the sessions table gets one row per
request and grouping is meaningless.

- [x] Accept optional `session_id` on both POST endpoints
- [x] Reuse when valid, create only when absent
- [x] Validate by existence in the `sessions` table, not just by UUID parse.
      `session.validate_session` only checks that the string parses, so a
      well-formed but unknown id would pass and then violate the foreign key on
      `inference_requests.session_id`. Fall back to creating a new session when
      the id is well formed but unknown. Implemented as
      `SessionTracker.resolve_session`, backed by `db.session_exists`.
- [x] Frontend: store session id in React state, send on every request. Held in a
      `SessionProvider` context above the router so Analyze and Batch share one id.
      Both endpoints echo `session_id` back in the response.
- [x] **Verify:** the live database shows seven consecutive requests sharing a
      single session id, where the old code would have written seven rows into
      `sessions`. Unit checks also confirm a well-formed but unknown id is
      refused, and that the foreign key would have rejected it (IntegrityError).

### 1.5 Truncated file validation

`img.verify()` reads headers only, so a truncated JPEG passes and then crashes
during preprocessing.

- [x] Replace `verify()` with `load()` in `pipeline.validate_image`
- [x] Replace `verify()` with `load()` in `batch_handler.validate_file`
- [x] **Verify:** truncated JPEG (header intact, data cut in half) now returns
      False from both validators where `verify()` returned True, so `/api/analyze`
      rejects it with a 400 before `preprocess` is reached. Checked at unit level
      on 27 July 2026; still worth one manual upload during the Phase 2.7 pass.

### 1.6 Probability precision

`consensus_score` and `predicted_probability` were `Numeric(5, 2)`, which
truncates to 2dp. We store `round(probability, 4)`, so 0.8734 became 0.87 and
History disagreed with Analyze on the same image.

- [x] Change both columns to `Numeric(6, 4)`, verified 27 July 2026 in
      `database.py:33` and `database.py:42`, and against the live `ai_detection`
      database, which reports `numeric(6,4)` on both columns

**Deployment caveat, still live.** Because the column type changed, an existing
database must have its tables DROPPED, not truncated. `create_all` does not alter
existing tables. On any database created before this change, including a marker's
fresh setup or a second machine, run:

```sql
DROP TABLE model_outputs, inference_requests, batches, sessions CASCADE;
```

then restart the backend so `create_all` rebuilds them. Not required on the
current local database, which already has the right types and holds zero rows.

### 1.7 Housekeeping

- [ ] Delete `batch_handler.get_batch_summary`, a dead duplicate of
      `results.format_batch_summary`. Keep the `results.py` version, which is the
      one Phase 3.1's batch summary test should target.
- [x] `results.py`: `target_layer` was read before the `model is not None` guard,
      so the None branch that guard exists for would have raised AttributeError
      first. Moved inside the guard while fixing 1.3.
- [x] `RUN.md`: weights path said `backend/models/checkpoints/`, actual location
      is `backend/models/weights/`. Would have broken Phase 6.3, which follows the
      README literally.

---

## Phase 2: Backend implementation

Three to four days. This is what turns "partially integrated" into a working
system, and it is what the recorded demonstration depends on.

### 2.1 Input validation

- [ ] Max file size on `/api/analyze` (10MB, matching the zip limit), returns 413
- [ ] Extension and MIME check on `/api/analyze`, returns 415
- [ ] `model_name` as a FastAPI `Enum` so bad values give an automatic 422
- [ ] Reject empty uploads, returns 400

### 2.2 Error handling

- [ ] Wrap `preprocess` and `predict` in try/except, return a clean message
- [ ] Wrap the two `db.save_*` calls so a database failure does not discard a
      successful inference (return the result with a warning flag)
- [ ] Add `logging` and log every caught exception with context
- [ ] Global exception handler so no stack trace ever reaches the client

### 2.3 Zip hardening

- [ ] Check `zf.getinfo(name).file_size` before calling `zf.read`
- [ ] Cap total files per zip (suggest 100)
- [ ] Cap total uncompressed size (suggest 200MB)
- [ ] Handle `zipfile.BadZipFile`, returns 400
- [ ] Reject a zip containing zero valid images, returns 400 with a useful message

### 2.4 Batch lifecycle

- [ ] Mark batch status complete after processing
- [ ] Mark batch status failed on exception
- [ ] Record processed and skipped counts

### 2.5 Health endpoint

- [ ] Report models-loaded state per model
- [ ] Ping the database
- [ ] Report device (cuda or cpu)
- [ ] Return 503 when anything is down

### 2.6 Concurrency

- [ ] Change endpoints from `async def` to `def` so FastAPI uses a threadpool
- [ ] Measure single-request latency before and after
- [ ] Note both numbers for the report (relevant to the 3000ms NFR)

### 2.7 End-to-end verification

- [ ] Upload, then inference, then database write, then visible in History (single)
- [ ] Upload, then inference, then database write, then visible in History (batch)
- [ ] All four models work through the UI
- [ ] Grad-CAM renders for CNN and Hybrid
- [ ] Spectrogram renders for FFT and Hybrid
- [ ] Feature importance renders for STM
- [ ] Search and category filter both work
- [ ] **Screenshot everything.** These become report figures and demo material

### 2.8 Visualisation credibility

From live testing on 27 July 2026. A Minecraft screenshot returned 0.3% P(AI)
from CNN, 0.1% from Hybrid, 25.2% from STM and 68% from FFT. Three models call it
authentic; FFT disagrees because blocky repeated textures produce periodic
frequency signatures. Contrast 19.8 and sensor noise 5.4 are both correct
readings for a render, but the panels labelled them "below natural range" and
"low camera fingerprint", which reads as evidence of AI origin when it is not.

#### 2.8.1 Split the results panel into two labelled groups

- [ ] "What this model analysed" holds only evidence the selected model actually
      used: Grad-CAM for CNN and Hybrid, spectrogram for FFT and Hybrid, feature
      contributions for STM
- [ ] "Image properties" holds contrast, sensor noise and RGB, labelled as general
      characteristics computed from the image rather than model inputs
- [ ] Exception: for STM, noise residual and colour statistics are genuine
      features, so they stay in the evidence group

#### 2.8.2 Neutralise the generic metric captions

- [ ] State what each number measures and its typical range, nothing more
- [ ] Remove every claim about what a value suggests regarding AI origin,
      including the three "what to look for" lines under the RGB histogram
- [ ] Remove the "Why do metrics differ from the verdict?" box, which exists only
      to explain away the claims being removed above

#### 2.8.3 Non-photographic content notice

- [ ] When contrast is below roughly 25 and sensor noise below roughly 8, show a
      note: the models are trained on photographs against AI-generated images, so
      renders, screenshots and vector graphics fall outside both training classes
      and the verdict is less reliable

#### 2.8.4 Spectrogram reference comparison

- [ ] Show the uploaded image's spectrum alongside two static references, one
      typical photograph and one typical diffusion output, three panels side by side
- [ ] Generate the two references once from existing dataset samples and ship them
      as static assets

#### 2.8.5 Multi-model comparison view (optional, decide later)

- [ ] Estimate the work before committing. `pipeline.predict_all` already exists,
      so a mode that runs all four and shows the verdicts together would surface
      model disagreement directly, which is the core research finding

### 2.9 Model selector labels

The selector currently calls FFT "Not Recommended" and Hybrid "Recommended".
Those labels describe in-distribution performance only. The core research finding
is the opposite for unseen modern generators: FFT detects 85.2% of MNW while
Hybrid detects 1.4% and CNN 2.2%. As written, the application contradicts the
dissertation and invites an obvious viva question.

- [ ] Reword so the basis of each label is explicit, for example "Best
      in-distribution accuracy" for Hybrid and "Best on unseen modern generators"
      for FFT
- [ ] Alternative: drop the recommendation language entirely and state each
      model's strength
- [ ] Same wording appears in both `AnalyzePage.jsx` and `BatchPage.jsx`

---

## Phase 3: Testing

One week. Currently scored 6.5/10 because Chapter 6 evaluates models but never
tests the software. This is a named gap in your feedback.

### 3.1 Unit tests (`pytest`)

- [ ] `test_pipeline.py`: validate_image accepts valid, rejects corrupt and truncated
- [ ] `test_pipeline.py`: preprocess output shape and dtype
- [ ] `test_pipeline.py`: predict returns all expected keys
- [ ] `test_pipeline.py`: unknown model name raises
- [ ] `test_batch_handler.py`: extract_zip skips non-images and oversized entries
- [ ] `test_batch_handler.py`: process_batch records per-file errors without aborting
- [ ] `test_results.py`: probability zone boundaries (0.2, 0.4, 0.6, 0.8)
- [ ] `test_results.py`: batch summary maths, including the empty case
- [ ] `test_session.py`: UUID validation accepts and rejects correctly

### 3.2 API tests (`TestClient`)

- [ ] `/api/health` returns 200 with expected shape
- [ ] `/api/analyze` happy path for each of the four models
- [ ] `/api/analyze` rejects oversized file (413)
- [ ] `/api/analyze` rejects wrong type (415)
- [ ] `/api/analyze` rejects unknown model (422)
- [ ] `/api/batch` happy path
- [ ] `/api/batch` rejects zip with no valid images (400)
- [ ] `/api/batch` rejects malformed zip (400)
- [ ] `/api/history` returns rows, search filters, category filters

### 3.3 NFR verification

Each NFR needs a stated test and a pass criterion.

- [ ] NFR accuracy: F1 above 0.75 on Stage 3 (already evidenced, cite the table)
- [ ] NFR latency: verdict within 3000ms, measure per model, tabulate
- [ ] NFR replaceability: swap a checkpoint, confirm no other file changes
- [ ] NFR CPU deployability: run the full app with CUDA disabled

### 3.4 Test plan document

- [ ] Table: test ID, what is tested, method, pass criterion, result
- [ ] Map each test back to its FR or NFR
- [ ] Note coverage percentage if you run `pytest-cov`
- [ ] This becomes a new section in Chapter 6

---

## Phase 4: Hybrid fix and research completion

One week. Research is already A-grade, so this is upside rather than repair.

### 4.1 Feature imbalance

CNN contributes 2048 features against FFT's 256, so the fusion head learns to
lean on CNN and suppresses the spectral signal.

- [ ] Add L2 normalisation to both branches before concatenation
- [ ] Retrain Hybrid at Stage 3 only
- [ ] Re-run OOD tests (MNW and Chameleon, all degradation levels)
- [ ] Compare against the current numbers

Either outcome is a result worth reporting. If MNW detection jumps, that is a
headline finding. If it does not, "scale normalisation alone is insufficient,
the imbalance is learned rather than purely dimensional" is equally publishable.

- [ ] If normalisation alone fails, try projecting CNN 2048 down to 256 so both
      branches are literally equal size, then retrain and retest

### 4.2 Statistical validity

Feedback flagged single-seed results as not statistically defensible, given you
claim CNN and Hybrid are "effectively tied".

- [ ] Re-run Stage 3 with seeds 1 and 2 for all four models (overnight)
- [ ] Report mean and standard deviation
- [ ] If time runs short, state the limitation explicitly instead

### 4.3 Optional, if time allows

- [ ] Unseen GAN family test set (StyleGAN3 or ProGAN) to complete the OOD picture
- [ ] Ensemble voting across all four models, hard and soft

---

## Phase 5: Report fixes

One week. Every item below comes directly from the interim feedback. This phase
carries the most marks per hour of any work remaining.

### 5.1 Requirements (biggest single gap: 16.5/25)

- [ ] Complete FR5, currently truncated mid-sentence on p.12
- [ ] Give every FR an ID (FR1, FR2, …)
- [ ] Add MoSCoW priority to each FR
- [ ] Add FRs for error handling, supported file types, size limits
- [ ] Add FRs for history search and filter behaviour
- [ ] Add an FR for history record display: every stored record shows P(AI)
      alongside the verdict. Currently unspecified, which is how the missing
      column in `HistoryPage.jsx` went unnoticed
- [ ] Add a behavioural requirements table linking each use case to its FRs

### 5.2 Design (17.5/25)

- [ ] Re-export the class diagram at full page width (currently unreadable, p.18)
- [ ] Re-export the Gantt chart (unreadable, p.57)
- [ ] Cut the FAQ activity diagram (linear, no decision points, adds nothing)
- [ ] Add a traceability table: FR to diagram to class that satisfies it

### 5.3 Introduction and literature review (6/10)

- [ ] Consolidate five "aims" into one or two, move the rest into objectives
- [ ] Add a synthesis section positioning the work against the wider field
- [ ] Actually discuss Wang et al. (2020), currently reference-list only
- [ ] Actually discuss Yan et al. (2024), currently reference-list only
- [ ] Target 10 to 15 discussed sources, not four

### 5.4 Implementation

- [ ] Replace "Grad-CAM has been architected" with a real screenshot
- [ ] Update Chapter 5 to state the backend is complete
- [ ] Update Chapter 7 progress section to match reality

### 5.5 Report quality (6.6/10)

- [ ] Change "inspired with DEFEND" to "inspired by" (section 3.3)
- [ ] Fix "approach each exploit" (section 1.3.1)
- [ ] Fix the truncated FR5 sentence (p.12)
- [ ] Remove the large blank block on p.15
- [ ] Full proofreading pass, read aloud or use a text-to-speech tool
- [ ] Check every figure is legible at print size
- [ ] Update TOC, list of figures, list of tables

### 5.6 New content

- [ ] Chapter 6: application test plan section
- [ ] Chapter 6: updated Hybrid results if the fix worked
- [ ] Chapter 6: multi-seed variance if completed
- [ ] Chapter 7: revise future work to reflect what got done
- [ ] State as a limitation that the system separates photographs from
      AI-generated images, and that synthetic non-AI content such as game renders
      sits outside both training classes. Use the Minecraft result as the worked
      example (CNN 0.3%, Hybrid 0.1%, STM 25.2%, FFT 68% P(AI)), noting FFT's
      sensitivity to periodicity as the likely cause of its disagreement. See 2.8.

---

## Phase 6: Final deliverables

One week, plus the first week of September as buffer.

### 6.1 Final report

- [ ] All Phase 5 items complete
- [ ] Word count within limit
- [ ] References complete and consistently Harvard
- [ ] Appendices: Gantt, supervisor logs, source code, slides
- [ ] Export to PDF, check figures survived the export

### 6.2 Source code as Word file

- [ ] Script the dump: every `.py` and `.jsx` with a filename header
- [ ] Monospace font, syntax readable
- [ ] Table of contents by file
- [ ] Check nothing is truncated

### 6.3 Executable

- [ ] Fresh clone on a clean machine or a fresh virtual environment
- [ ] Follow your own README exactly, fix anything that fails
- [ ] Confirm weight download link works
- [ ] Confirm database setup steps work from scratch

### 6.4 Recorded viva and demonstration

- [ ] Write the demo script (what you show, in what order)
- [ ] Rehearse the full run once, timing it
- [ ] Record: analyse a real image, analyse an AI image, run a batch, show history
- [ ] Show at least one Grad-CAM, one spectrogram, one feature importance chart
- [ ] Explain the key finding on camera (FFT reversal)
- [ ] Check audio before recording the real take
- [ ] Watch it back once before submitting

---

## Weekly summary

| Week | Dates | Focus |
|---|---|---|
| 1 | 26 Jul to 1 Aug | Phase 0, 1, 2: repo, bugs, backend |
| 2 | 2 to 8 Aug | Phase 3: testing |
| 3 | 9 to 15 Aug | Phase 4: Hybrid fix, reruns |
| 4 | 16 to 22 Aug | Phase 5: report fixes |
| 5 | 23 to 31 Aug | Phase 6: assembly, recording |
| Buffer | 1 to 5 Sep | Overflow and final checks |

---

## Where the marks are

| Area | Weight | Current | Realistic target |
|---|---|---|---|
| Requirements | 25% | 66% | 80% |
| Design | 25% | 70% | 82% |
| Implementation | 20% | 82% | 88% |
| Testing | 10% | 65% | 85% |
| Intro and lit review | 10% | 60% | 78% |
| Report quality | 10% | 66% | 85% |

Requirements and Design together are half the total mark, and both improve
through documentation work rather than new engineering. Do not leave Phase 5
until the last week.
