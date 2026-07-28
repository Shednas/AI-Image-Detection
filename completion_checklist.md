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
      - The two copies of the model files have diverged. `backend/models/stm_model.py`
        gained an `extract_features` method in Phase 1.8, which the research copy
        does not have. Keep the app copy when merging. Diff the other three
        before assuming they still match.
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
      `SessionProvider` context above the router so Analyze and Batch share one id,
      and mirrored into `sessionStorage` so a refresh mid-demo does not fragment
      the grouping. Both endpoints echo `session_id` back in the response.
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

- [x] Delete `batch_handler.get_batch_summary`, a dead duplicate of
      `results.format_batch_summary`. Keep the `results.py` version, which is the
      one Phase 3.1's batch summary test should target.
- [x] `results.py`: `target_layer` was read before the `model is not None` guard,
      so the None branch that guard exists for would have raised AttributeError
      first. Moved inside the guard while fixing 1.3.
- [x] `RUN.md`: weights path said `backend/models/checkpoints/`, actual location
      is `backend/models/weights/`. Would have broken Phase 6.3, which follows the
      README literally.

### 1.8 STM feature contributions are not per-image

Found in live testing. `results.generate_feature_importance` reads
`model.lgbm_model.feature_importances_`, which is a global property of the
trained model, so every image returns the same breakdown. Confirmed: a Minecraft
screenshot and a Gemini-generated image both gave HOG 86.5, LBP 7, Colour 3.3,
DCT 2.3, Noise 1. The caption claims the chart shows how much each group
influenced this prediction, which is false.

- [x] Use LightGBM per-prediction contributions:
      `lgbm_model.predict(features, pred_contrib=True)` returns SHAP values per
      feature plus a trailing bias term. Sum absolute contributions within each
      group in `FEATURE_GROUP_SLICES`, drop the bias term, normalise to
      percentages. Absolute rather than signed, because a group arguing against
      the verdict still influenced it and signed sums cancel within a group.
- [x] `STMDetector.extract_features` exposes the vector that `forward` used to
      compute and discard, so the contributions are taken from the same features
      the classifier scored rather than a second extraction
- [x] Reword the caption to match what the chart now shows
- [x] **Verify:** three structurally different images (gradient, noise, tiled
      blocks) give three different breakdowns, HOG ranging 62.0 to 89.8 and LBP
      5.7 to 21.5. The old code returned HOG 86.5, LBP 7, DCT 2.3, Colour 3.3,
      Noise 1 for all three, matching the values seen in live testing. Feature
      vector is 1822 long and `FEATURE_GROUP_SLICES` covers exactly 1822, with
      the 1823rd contribution column dropped as the bias term. Checked
      27 July 2026.
- [x] ~~Fallback to global importance~~ Not needed, the real fix works.
- [ ] Group sizes are very unequal, HOG being 1764 of the 1822 features, so HOG
      dominates most group totals. That is expected rather than a defect.
      Handled in 2.8.2, not here.

---

## Phase 2: Backend implementation

Three to four days. This is what turns "partially integrated" into a working
system, and it is what the recorded demonstration depends on.

### 2.1 Input validation

- [x] Max file size on `/api/analyze` (10MB, matching the zip limit), returns 413.
      Imports `MAX_FILE_SIZE_BYTES` from `batch_handler` rather than restating
      the number, so the two limits cannot drift apart.
- [x] Extension and MIME check on `/api/analyze`, returns 415. `content_type` is
      client-supplied, so this only screens obvious mismatches; the PIL decode in
      `validate_image` is still what proves the bytes are an image.
- [x] `model_name` as a FastAPI `Enum` so bad values give an automatic 422.
      `ModelName` lives in `pipeline.py` beside `MODEL_DISPLAY_NAMES`. Applied to
      `/api/batch` as well: an unknown model there previously reached
      `process_batch` and came back as a per-file error on every file, so the
      response looked like a completed batch rather than a rejected request.
- [x] Reject empty uploads, returns 400
- [x] **Verify:** all six rejection paths return the intended status through
      `TestClient` (empty 400, oversized 413, bad extension 415, bad MIME 415,
      unknown model 422, undecodable bytes 400), and a valid request still
      reaches inference. Checked 27 July 2026.

### 2.2 Error handling

- [x] Wrap `preprocess` and `predict` in try/except, return a clean message
- [x] Wrap the two `db.save_*` calls so a database failure does not discard a
      successful inference (return the result with a warning flag). Both
      endpoints now return `warning`, null when everything saved. The batch loop
      counts unsaved rows rather than aborting, and skips the rows entirely when
      the parent batch row failed, since they carry `batch_id` as a foreign key.
- [x] Add `logging` and log every caught exception with context. Replaced the two
      `print` calls in `results.py` at the same time.
- [x] Global exception handler so no stack trace ever reaches the client
- [x] **The dotenv working-directory bug does not reproduce.** `load_dotenv()`
      resolves through `find_dotenv()`, which walks up from the calling file
      rather than from the working directory, so `backend/.env` is found from any
      cwd including `C:\`. Verified 27 July 2026 by importing the module from
      both. The SETUP.md note about running uvicorn from `backend/` is about the
      `No module named 'database'` import error, which is a separate and real
      problem. Two genuine defects were there and are fixed:
      - `find_dotenv` walks all the way to the filesystem root, so on a clone
        with no `.env` yet it would silently adopt an unrelated one from a parent
        directory. `load_dotenv` is now anchored to `backend/.env`.
      - A missing `DATABASE_URL` reached `create_engine(None)` and raised a
        SQLAlchemy `ArgumentError` that never mentions the missing file. It now
        raises a message naming the file to create. This is the first thing a
        fresh clone hits, so it matters for Phase 6.3.
      - Note for 6.3: `find_dotenv` switches to the working directory under a
        frozen build, so the original concern becomes real if the executable is
        ever built with PyInstaller. The anchored path removes that too.
- [x] **Verify:** eleven checks through `TestClient` with stubbed pipeline and
      database. A failing save returns 200 with the result and a warning, a
      healthy save leaves the warning null, an inference failure returns a clean
      500, an unhandled error returns the generic message, neither leaks a
      traceback or the original exception text, and a missing `DATABASE_URL`
      raises a message naming the file. Checked 27 July 2026.
- [ ] Not done here: the frontend does not display `warning` yet, so a failed
      save is currently silent to the user. Wire it into the Analyze and Batch
      pages during 2.7.

### 2.3 Zip hardening

- [x] Check the declared size before reading, so a bomb is refused without being
      decompressed at all. The read is also bounded to one byte past the limit.
      That bound turned out to be defence in depth rather than the main guard:
      zipfile stops decompressing at the declared size by itself, and an entry
      that understates its size fails the CRC check instead.
- [x] Cap total files per zip, set to 100
- [x] Cap total uncompressed size, set to 200MB, returns 413
- [x] Handle `zipfile.BadZipFile`, returns 400. Two places, not one: the
      constructor for an unreadable archive, and the per-entry read for a
      damaged member. Only the first was obvious. A tampered or corrupt entry
      raises `BadZipFile` from `read`, which escaped `extract_zip` entirely and
      reached the global handler as a 500. Damaged entries are now skipped the
      same way oversized ones are, so one bad member does not cost the caller
      the rest of the archive.
- [x] Reject a zip containing zero valid images, returns 400 with a useful
      message. Four distinct messages: no images at all, all oversized, all
      damaged, or a mix with counts.
- [x] Rejections raise `ZipRejected`, which carries its own status code, so the
      endpoint maps the reason to a response without restating it
- [x] **Verify:** sixteen checks at unit and endpoint level. Covers the happy
      path, directory entries and non-images ignored, nested paths flattened,
      unreadable archive, empty upload, no images, file count cap at and over
      the limit, total size cap at and over the limit, an oversized entry, an
      entry with a falsified header size, and a damaged entry alongside a good
      one where the good one still comes through. Checked 27 July 2026.

### 2.4 Batch lifecycle

- [x] Mark batch status complete after processing. `BatchStatus` in
      `database.py` holds the three states; `save_batch` opens the row as
      `processing` and the endpoint closes it as `completed`.
- [x] Mark batch status failed on exception. `process_batch` traps per-file
      errors, so reaching that handler means the run collapsed as a whole,
      which is recorded as nothing processed and everything skipped.
- [x] Record processed and skipped counts, as `processed_files` and
      `skipped_files` on `batches`.
- [x] Schema change applied by `DatabaseManager._migrate`, which runs
      `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on every startup rather than
      relying on `create_all`. Nothing is dropped and the statements are
      idempotent, so a fresh clone and the existing database converge.
- [x] Existing rows backfilled rather than left to take the column default.
      They are finished test batches, so they are set to `completed`, while the
      default for new rows is `processing`. The counts are recovered from
      `inference_requests`, which holds one row per image that came through, so
      `skipped` is the remainder of `total_files`. A historical batch whose rows
      failed to save would be understated, but zero would misreport every batch
      rather than one.
- [x] **Verify:** twenty-two checks against a throwaway database holding the
      pre-change schema, plus endpoint-level checks with a recording stub.
      Covers the backfill, the derived counts, a batch with no rows, a batch
      with more rows than `total_files`, the NOT NULL and default state of all
      three columns, a second startup leaving the backfill untouched, a fresh
      database reaching the same shape, the completed and failed marks with
      their counts, and a batch row that failed to insert never being updated.
      Checked 28 July 2026.

Not done here: nothing user-facing reads `status` yet, so a failure to update it
is logged but raises no warning to the caller. Revisit if the History page ever
shows batch state.

### 2.5 Health endpoint

- [x] Report models-loaded state per model, from `pipeline.model_status()`
- [x] Ping the database with a `SELECT 1` round trip rather than a look at the
      connection pool, which would still call a dead server healthy
- [x] Report device (cuda or cpu)
- [x] Return 503 when anything is down, with `status` reading `degraded`
- [x] `load_models` now loads each model independently. It previously raised on
      the first missing checkpoint, which stopped the server before it could
      report which one was missing, so per-model health state was unreachable.
      Failures are logged with their reason and the service starts degraded.
- [x] `/api/analyze` and `/api/batch` return 503 for a model that is not loaded.
      On batch this previously produced an error on every file and looked like a
      completed batch, the same shape of problem the `ModelName` enum fixed in
      2.1. The two endpoints order the check differently on purpose: analyze
      validates the upload first, since that is one cheap decode and it gives the
      caller the more actionable error, while batch checks the model first, since
      opening the archive can cost the whole decompression limit and none of that
      work survives a 503.
- [x] `predict` no longer reports an unloaded model as an unknown name. That one
      branch covered both cases and sent anyone reading the log looking for a
      typo that was not there. Unknown names raise `ValueError`, unloaded models
      raise `ModelUnavailable`.
- [x] **Verify:** eighteen checks. Covers a pipeline with no weights on disk,
      each load failure being recorded, the two exception types, a healthy 200
      body, a 503 for one missing model, a 503 for an unreachable database, the
      absence of local paths in the response, and 503 from both endpoints when
      the requested model is not loaded. Checked 28 July 2026.

- [x] A database that is down at startup no longer stops the backend. This is
      the likeliest real failure, PostgreSQL simply not running yet, and the one
      most likely to spoil a recorded demo. `create_engine` never connects, so
      only `create_all` and the migration needed guarding. They now run behind
      `_ensure_schema`, which is attempted at startup, retried by `ping` on the
      first successful connection, and called by every session so no caller ever
      works against a schema that was never created. The backend starts degraded
      and recovers on its own when the database comes back, without a restart.
- [x] **Verify:** the backend was started against an unreachable database and
      answered `/api/health` with 503 and `"database":"down"` while all four
      models reported `true`, with the process still alive. Recovery was checked
      separately across ten checks: construction against a dead database does
      not raise, `ping` reports down, a write still fails so the caller keeps its
      warning, then once the database is reachable `ping` reports up, the four
      tables and the three new columns are created on that first ping, and
      writes succeed without a restart. Checked 28 July 2026.

Not done here: the load failure reasons stay in the log and are not returned,
since they carry local filesystem paths.

Not verified by stopping the PostgreSQL service, which needs an elevated shell.
An unreachable address gives the backend the same connection refused error, so
the behaviour under test is identical, but if you want the literal check run
`Stop-Service postgresql-x64-18 -Force` as administrator, start the backend, and
confirm the same 503.

### 2.6 Concurrency

- [x] Change endpoints from `async def` to `def` so FastAPI uses a threadpool.
      `/api/analyze`, `/api/batch` and `/api/history` are now `def`. The two
      upload endpoints read `file.file.read()`, since `await file.read()` is the
      async API. `/api/health` was already `def`.
- [x] Measure single-request latency before and after
- [x] Note both numbers for the report (relevant to the 3000ms NFR)

Single-request medians, seven timed runs each after two warm-ups, one fixed
512x512 PNG, one reused session, measured against a real uvicorn server on CUDA.

| Model  | Before (async) | After (def plus lock) |
|---|---|---|
| CNN    | 222.5 ms | 166.0 ms |
| FFT    | 134.5 ms | 124.7 ms |
| Hybrid | 174.9 ms | 216.1 ms |
| STM    | 236.0 ms | 222.2 ms |

Read those as unchanged, not improved. Run to run variance across repeats of the
same code was wider than the differences here: hybrid measured 174.9, 281.0 and
216.1 ms in three runs. The honest claim is that moving to a threadpool costs
nothing on a single request. Every model sits an order of magnitude under the
3000ms NFR, STM closest at 222 ms, which is the number Phase 3.3 needs.

Four concurrent requests, the case the change was meant to help:

| | Before (async) | After (def plus lock) |
|---|---|---|
| 4 back to back | 933.7 ms wall | 975.3 ms wall |
| 4 at once | 3336.8 ms wall | 2770.9 ms wall |
| slowest of the 4 | 2986.7 ms | 2430.6 ms |

- [x] Concurrency improved by about 17 per cent, which is far less than the
      change promises. Four simultaneous requests are still slower in total than
      the same four run back to back, so this is not parallelism, it is reduced
      queueing. A single request in a four-way pile-up takes about 2.4 s, inside
      the 3000ms NFR but without much room. Worth stating plainly in the report
      rather than claiming the threadpool fixed concurrency.
- [x] Torch thread oversubscription was tested as the explanation and ruled out.
      Torch asks for 14 intra-op threads per request against 20 logical cores, so
      four requests want 56. Capping with `OMP_NUM_THREADS=4` made both single
      request and concurrent numbers worse, so that is not the bottleneck. The
      remaining cause is not identified.
- [x] The change exposed a real correctness bug, not just a performance one.
      Grad-CAM registers hooks on a layer of the one shared model instance. Once
      requests genuinely overlapped, a hook registered by one request fired
      during another request's forward pass and overwrote the activations the
      first was about to read. Measured: 5 of 24 concurrent requests returned a
      heatmap belonging to a different image, and 1 returned none. Under
      `async def` this could not happen, because requests never overlapped.
- [x] Fixed with one reentrant lock on the pipeline, taken by `predict`, by
      Grad-CAM and by the STM contribution call. A lock inside Grad-CAM alone was
      tried first and left 1 of 24 still wrong, because an ordinary `predict`
      call passes through the same hooked layer. `ResultsHandler` is given the
      pipeline's lock rather than owning one, since the section that has to be
      serialised spans both modules. This also means model passes are serialised
      by design, which accounts for part of why concurrency does not scale. The
      GPU serialises them anyway.
- [x] **Verify:** the contamination test compares each concurrent response
      against a sequential reference for the same image, having first confirmed
      the two references are stable and different from each other. 24 concurrent
      Grad-CAM requests and 24 concurrent STM requests all match their own image.
      The 2.3, 2.4 and 2.5 suites were re-run against the changed code.
      Checked 28 July 2026.

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
- [ ] STM feature contributions: show the group total and a per-feature mean side
      by side. They answer different questions. The group total says what
      contributed most to this decision overall, the per-feature mean says which
      feature type is most informative individually. HOG dominating the total is
      expected given it is 1764 of the 1822 features, so the caption should say
      that rather than treat it as a finding. See 1.8.

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
- [ ] `httpx` was installed into `backend/venv` by hand on 27 July 2026 because
      `TestClient` needs it. It is in no requirements file, so a fresh clone
      cannot run these tests. Decide when starting this section whether it goes
      in `app/backend/requirements.txt` or a separate `requirements-dev.txt`, and
      add it either way. Do not leave it undocumented. Note that Starlette now
      warns on every `TestClient` import that `httpx` is deprecated in favour of
      `httpx2`, so pin whichever one still works when this section is written.

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
- [ ] Worked example on a post-cutoff generator: a Gemini Flash 3.6 image, a
      generator absent from the training data, gave CNN 0.1% P(AI), Hybrid 0.0%,
      STM 38.1% and FFT 53.9%. Only FFT reached the correct verdict, and only at
      borderline confidence. Present it as an illustration of the MNW
      generalisation finding rather than as evidence in its own right, since
      n=1.

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
