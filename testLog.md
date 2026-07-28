# Test log

One entry per verification run. The columns map onto the Phase 3.4 test plan
table, so nothing here needs rewriting later.

Entries T-001 to T-014 are backfilled from the scratchpad checks written during
Phase 2 on 27 and 28 July 2026. Those checks were throwaway harnesses, not the
pytest suite; Phase 3.1 and 3.2 salvage them into permanent tests.

Format:

    ## T-000 Short title
    Date:
    Area: which checklist item
    What could break: one line
    Method: how it was checked
    Result: pass or fail, with the detail

---

## T-001 Upload rejection paths return the right status
Date: 27 July 2026
Area: 2.1 input validation
What could break: a bad upload reaches PyTorch and comes back as a 500, telling
the user nothing about what was wrong with their file.
Method: six requests through `TestClient`, one per rejection path, plus a valid
request to confirm the happy path still reaches inference.
Result: Pass. Empty 400, oversized 413, bad extension 415, bad MIME 415, unknown
model 422, undecodable bytes 400.

## T-002 Unknown model name is rejected before inference
Date: 27 July 2026
Area: 2.1 input validation
What could break: an unknown model reaches `predict` and returns 500 on analyze,
or on batch returns an error on every file so the response looks like a
completed batch rather than a rejected request.
Method: posted `model_name=nonsense` to both endpoints.
Result: Pass. 422 from both, raised by the `ModelName` enum before the handler
body runs.

## T-003 STM feature contributions differ per image
Date: 27 July 2026
Area: 1.8 STM per-image contributions
What could break: the chart claims to explain this image while showing a global
property of the trained model, identical for every upload.
Method: ran three structurally different images through
`generate_feature_importance` and compared the group breakdowns. Also confirmed
the old code reproduced the reported constant values.
Result: Pass. Old code gave HOG 86.5 / LBP 7 / DCT 2.3 / Colour 3.3 / Noise 1 for
every image. New code gave HOG 62.0 to 89.8 and LBP 5.7 to 21.5 across the three.

## T-004 dotenv is not working-directory dependent
Date: 27 July 2026
Area: 2.2 error handling
What could break: the backend loads no `DATABASE_URL`, or silently adopts an
unrelated `.env` from a parent directory.
Method: imported `database.database` from the repo root and from `C:\`, printing
the resolved path and whether `DATABASE_URL` was set.
Result: Pass, and the premise was wrong. `find_dotenv` walks up from the calling
file, not the working directory, so the reported bug did not reproduce. Two real
defects were found instead and fixed: an unanchored search path, and a
`create_engine(None)` error that never mentions `.env`.

## T-005 Zip extraction rejects hostile and malformed archives
Date: 27 July 2026
Area: 2.3 zip hardening
What could break: a zip bomb exhausts memory, a crafted path escapes the
extraction directory, or one damaged member costs the caller the whole archive.
Method: sixteen checks at unit and endpoint level. Happy path, directory entries
and non-images ignored, nested paths flattened, unreadable archive, empty upload,
no images, file count cap at and over the limit, total size cap at and over the
limit, an oversized entry, an entry with a falsified header size, and a damaged
entry alongside a good one.
Result: Pass, after two real findings. `BadZipFile` raised from `read()` escaped
`extract_zip` entirely and became a generic 500; damaged entries are now skipped
like oversized ones. The 413 message also printed a stale limit from a second
constant, now derived at message time.

## T-006 A falsified zip header cannot exceed the size limit
Date: 27 July 2026
Area: 2.3 zip hardening
What could break: an archive that understates its uncompressed size in the header
gets past the declared-size check and decompresses without bound.
Method: built a 12MB entry, patched both headers to claim 1000 bytes, and
extracted it.
Result: Pass, and it corrected a wrong assumption in the code comments. zipfile
itself stops at the declared size and fails the CRC, so the entry is rejected as
damaged. The declared-size check is the effective guard and the bounded read is
defence in depth, not the other way round.

## T-007 Batch lifecycle status and counts
Date: 28 July 2026
Area: 2.4 batch lifecycle
What could break: a batch row stays at its initial state forever, so History
cannot distinguish a finished batch from one that died mid-run.
Method: twenty-two checks against a throwaway database holding the pre-change
schema, plus endpoint-level checks with a recording stub. Covers the completed
and failed marks with their counts, and a batch row that failed to insert never
being updated.
Result: Pass. Completed batches record processed and skipped counts; a collapsed
run records failed with nothing processed.

## T-008 Migration backfills historical batches correctly
Date: 28 July 2026
Area: 2.4 batch lifecycle
What could break: existing viva batches inherit the `processing` default and are
mislabelled as unfinished, or the migration drops or rewrites data.
Method: built the pre-change schema in a throwaway database, inserted three
historical batches with differing row counts, ran `DatabaseManager`, then
inspected every row and column definition. Repeated on a second startup and on a
clean database.
Result: Pass. Historical rows backfill to `completed` with counts derived from
`inference_requests`; a batch with no rows counts as all skipped; more rows than
`total_files` cannot produce a negative count. A second startup changes nothing.

## T-009 Live database migration
Date: 28 July 2026
Area: 2.4 batch lifecycle
What could break: the migration behaves differently against the real viva data
than against a synthetic copy.
Method: inspected `ai_detection` after the migration ran, listing all columns of
`batches` and every row.
Result: Pass. Four historical batches, all `completed`, `processed_files` 1 and
`skipped_files` 0 each, matching their single-image contents. The 16
`inference_requests` rows are untouched.

## T-010 Health endpoint reports real dependency state
Date: 28 July 2026
Area: 2.5 health endpoint
What could break: health returns 200 while a model failed to load or the database
is unreachable, so a broken deployment looks healthy.
Method: eighteen checks. A pipeline with no weights on disk, each load failure
being recorded, the two exception types, a healthy 200 body, a 503 for one
missing model, a 503 for an unreachable database, and 503 from both endpoints
when the requested model is not loaded.
Result: Pass. 200 with all four models `true` and `database: up`; 503 with
`status: degraded` otherwise. No local filesystem paths appear in the response.

## T-011 Backend starts with the database down
Date: 28 July 2026
Area: 2.5 health endpoint
What could break: PostgreSQL not running stops the backend from starting at all,
so health cannot report the failure. This is the likeliest real failure and the
one most likely to spoil a recorded demo.
Method: started uvicorn against an unreachable address and requested
`/api/health`.
Result: Pass. HTTP 503 with `database: down`, all four models `true`, process
still alive. Not verified by stopping the PostgreSQL service, which needs an
elevated shell; an unreachable address produces the same connection refused
error.

## T-012 Database recovery without a restart
Date: 28 July 2026
Area: 2.5 health endpoint
What could break: the backend never recovers after the database comes back,
because the schema was never created and nothing retries.
Method: ten checks. Constructed a manager against a dead database, confirmed
`ping` reports down and writes still fail, then pointed it at a reachable
database and pinged again.
Result: Pass. The schema and the migration are built on the first successful
ping, and writes succeed afterwards without restarting the process.

## T-013 Endpoint latency before and after the threadpool change
Date: 28 July 2026
Area: 2.6 concurrency
What could break: moving blocking endpoints to a threadpool slows down the single
request case, which is what the 3000ms NFR measures.
Method: seven timed runs per model after two warm-ups, one fixed 512x512 PNG, one
reused session, against a real uvicorn server on CUDA. Repeated before and after
the change, plus four requests back to back and four at once.
Result: Pass on the NFR, negative on the goal. Single-request medians differ by
less than the run-to-run variance on identical code, so the change costs nothing
but gains nothing there either. STM is closest to the limit at 222ms. Four
concurrent requests improved from 3337ms to 2771ms wall, about 17 per cent, but
the same four run back to back take 975ms, so this is reduced queueing rather
than parallelism. Torch thread oversubscription was tested with
`OMP_NUM_THREADS=4` and ruled out; it made both numbers worse.

## T-014 Grad-CAM under concurrent requests
Date: 28 July 2026
Area: 2.6 concurrency
What could break: two requests overlap inside the one shared model instance and a
user is shown a heatmap generated from somebody else's image.
Method: took sequential reference heatmaps for two visually distinct images,
confirmed they were stable across repeats and different from each other, then ran
24 concurrent requests alternating the two and compared each response against its
own reference. Repeated for STM feature contributions.
Result: Fail, then pass. On the first run 5 of 24 heatmaps belonged to a
different image and 1 came back empty. A lock inside Grad-CAM alone left 1 of 24
wrong, because an ordinary `predict` call passes through the same hooked layer.
With one reentrant lock on the pipeline, shared by `predict`, Grad-CAM and the
STM contribution call, 24 of 24 Grad-CAM and 24 of 24 STM responses match their
own image.
