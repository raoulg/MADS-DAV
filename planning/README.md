# planning/

> **Status lives in Linear**, not here: [DAV refactor](https://linear.app/pttrn/project/dav-refactor-01967b8561ae)
> — seven milestones (M0–M6), 42 issues. Work from the board.
>
> These files stay as the **design record**: the reasoning, the numbers, and what was tried
> and rejected. Linear says *what is next*; this says *why it is shaped that way*. When the
> two disagree about status, Linear wins.

The refactor of MADS-DAV, written down before any notebook is touched.

Each `.yml` file is one axis of the refactor. They are meant to be read in order, but each one
stands on its own — `05-notebooks.yml` is the file you open when you actually start editing.

| File | What it decides |
|---|---|
| `00-overview.yml` | Why this refactor, what stays, what is explicitly out of scope |
| `01-showcase-principle.yml` | The two-half notebook pattern that replaces "hope the student's chat shows it" |
| `02-datasets.yml` | Which dataset showcases which visualisation type, and why |
| `03-goad-integration.yml` | Where `goad_toolkit` enters each lesson, and what goad needs to grow |
| `04-mcp-integration.yml` | The goad MCP server: tools, staged coaching, course-wide assistant policy |
| `05-notebooks.yml` | Per-notebook: current state → target state → concrete edits |
| `06-huggingface-migration.yml` | `mads_datasets` → HuggingFace, and the bridge to the ML course |
| `07-goad-exercises-reuse.yml` | What to lift from `goad_exercises` and where it lands |
| `08-learning-goals.yml` | Changes to `references/leerdoelen/` and the rubric |
| `09-milestones.yml` | Execution order, with what blocks what |
| `10-spurious-findings.yml` | The three "aha" moments that stop students inventing findings |

Status vocabulary used throughout: `todo`, `doing`, `done`, `proposed` (needs a yes from Raoul
before it becomes work), `verify` (a factual claim to check before relying on it).
