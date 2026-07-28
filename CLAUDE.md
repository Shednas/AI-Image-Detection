# CLAUDE.md
Run claude --continue
## Project

Dissertation comparing four AI image detection models: CNN, FFT, Hybrid and STM.
Final submission 5 September 2026. The interim submission scored 70/100.

`completion_checklist.md` is the authoritative task list. Work from it, and keep
it in step with reality as items land.

This directory is the application, not the research lab.

## Critical convention: P(AI) versus P(real)

Training used `{ai_generated: 0, real: 1}`, so the raw sigmoid output is
**P(real)**, not P(AI).

- `pipeline.predict` returns both `p_real` and `p_ai`.
- Everything user-facing shows `p_ai`.
- Never recompute `1.0 - p_real` anywhere. Read `p_ai` from the pipeline.

This has already caused one live bug: the batch endpoint wrote P(real) into the
same `consensus_score` column that the analyze endpoint filled with P(AI), so
one image gave contradictory numbers across pages.

## Writing conventions

- UK spelling.
- No em dashes anywhere, in code, comments, docs or chat. Use commas, colons or
  brackets.
- No AI-tell artifacts: no emojis, no character-drawn arrows in prose or
  comments, no decorative banner comments, no multiple spaces used to align code
  or comments.
- Comments explain why, not what.

## Comment density

Comment sparingly. A comment earns its place only when the reasoning behind a
line is genuinely not recoverable from reading it: the `{ai_generated: 0,
real: 1}` mapping, the P(AI) direction, the lock spanning `pipeline.py` and
`results.py`.

Do not write comments that:

- restate what the code does
- explain standard framework or library behaviour
- narrate the obvious, or label a section of ordinary code
- record how something used to work, unless the old behaviour was a bug someone
  could reintroduce

Prefer one line to three. If a comment needs a paragraph, the reasoning belongs
in `completion_checklist.md` or `testLog.md`, not beside the code.

## Data constraint

`inference_requests` holds results needed for the viva.

- Never DROP any table without asking first.
- Use `ALTER TABLE` for schema changes, including the `status` column in Phase
  2.4. `create_all` does not alter existing tables, so a new column needs an
  explicit ALTER.

## Working rules

- Always ask approval before running a shell command. Never use the "don't ask
  again" option.
- Keep reports short: a bullet list, one line per change.
- Explain properly only when there is a real decision to make, a user-visible
  behaviour change, or something to disagree with.
- One commit per checklist item.
- No `Co-Authored-By` trailer on commits.

## Status

Phase 1 is complete except the 1.1 cross-page verification, which is done
manually and is not a coding task.

Phase 0 (repo restructure into `research/` and `app/`, README files, licence,
weights release) is untouched apart from `git init` and `.gitignore`.

Next is Phase 2, in order:

1. 2.1 input validation
2. 2.2 error handling, including the dotenv working-directory bug
3. 2.3 zip hardening
4. 2.4 batch lifecycle
5. 2.5 health endpoint
6. 2.6 concurrency

Stop before 2.7. It is end-to-end verification and needs a person driving the UI.
