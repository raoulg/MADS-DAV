# Manual setup

This is the by-hand version of the setup. The recommended route is to hand
`references/tooling-setup.md` (and, if you have a lab VM, `references/lab-setup.md`) to
an assistant and let it walk you through, one step at a time. Use this page when you
have no assistant with MCP access, or when you want to see every command on one page.

All commands assume a Unix shell: Terminal on macOS, Git Bash on Windows.

## 1. uv

`uv` installs Python, manages the packages a project needs, and records the exact
versions in `uv.lock` so the environment can be rebuilt anywhere.

```bash
which uv
```

If that prints nothing, install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run the PowerShell line from the
[uv documentation](https://docs.astral.sh/uv/getting-started/installation/) once, then
go back to Git Bash. Open a new terminal afterwards.

Python itself comes from uv: `uv python install 3.12`. Do not use a python.org, brew,
Microsoft Store or conda Python for course work; a second interpreter next to uv's is the
usual cause of "it works in the terminal but not in the notebook".

The [uv guide on projects](https://docs.astral.sh/uv/guides/projects/) is ten minutes
well spent: it explains what `uv sync`, `uv run` and `uv add` do.

## 2. The repository

```bash
git clone https://github.com/raoulg/MADS-DAV.git
cd MADS-DAV
uv sync
```

`uv sync` creates `.venv` and installs exactly what `uv.lock` lists, including the CPU
build of torch for lesson 6 (a few hundred megabytes, once).

Then, and this is the rule of the course:

```bash
git checkout -b mywork
```

`main` belongs to the teacher and receives new material every week. Work on your own
branch, and **copy a notebook under a new name before editing it**
(`01.1-goad-toolkit-101.ipynb` → `01.1-goad-toolkit-101_<yourname>.ipynb`). A renamed
file never conflicts with an update to the original. To get updates:

```bash
git status            # must be clean: commit first
git checkout main
git pull
git checkout mywork
git merge main
```

Do not `uv add` to this repository's `pyproject.toml`; it is the teacher's file, and the
same conflict rule applies.

## 3. Editor

Open VS Code in the `MADS-DAV` folder itself, not its parent. Install the **Python** and
**Jupyter** extensions, open a notebook, and select the `.venv` kernel top right. If
`.venv` is not offered, reload the window; if it is still not offered, you opened the
wrong folder.

Recommended: the **Git Graph** extension (`mhutchie.git-graph`), a visual history of
branches and merges. Any editor is fine, provided it can edit over SSH; a later course
moves to a remote VM.

## 4. Pre-commit hooks

`.lefthook.yml` runs, before every commit: `ruff format` and `ruff check` (style and
lint), `ty` (types, on `.py` files), `jupyter nbconvert --clear-output` (notebooks are
committed without outputs), `notebooktester` (every notebook runs top to bottom), and
`lychee` (dead links). Activate them once per clone:

```bash
uv tool install lefthook
lefthook install
```

Notebooks are committed without outputs for two reasons: cell output is a cache, not a
place to store results (save figures and tables to files), and re-running a cell would
otherwise change the file and produce commits and conflicts about nothing. The
consequence is that a notebook must run top to bottom from a fresh kernel. Cells whose
runtime scales with a number get `from notebooktester import param` and
`EPOCHS = param(50, test=2)`, so the tester finishes in seconds.

`lychee` is the one hook that is not a Python package. Install it through your own
package manager (`brew install lychee` on macOS, `winget install --id lycheeverse.lychee`
on Windows, or a
[prebuilt binary](https://github.com/lycheeverse/lychee/releases)), or skip it on this
machine:

```bash
printf 'pre-commit:\n  commands:\n    lychee:\n      skip: true\n' > .lefthook-local.yml
```

To run the same checks by hand:

```bash
uv run ruff check .                          # lint
uv run python tools/check_pipeline_drift.py  # lesson 1's ParseIRCLines vs scripts/pipelines.py
uv run notebooktester -f -t 900 notebooks/   # every notebook, top to bottom
```

Every notebook is expected to run on the showcase data alone, with no `config.toml`
present; that is what CI checks on every push.

## 5. The coaches

`.mcp.json` registers two MCP servers: `goad` (is my analysis any good?) and
`codestyle` (is my code any good?). Each is one command, `uv run --no-project` on a
script fetched from GitHub at a pinned tag, so there is nothing to install. See
[CLAUDE.md](../CLAUDE.md) for what they expect of you: they coach, they do not answer.

- **Claude Code**: open the folder; it offers to connect the project's servers and you
  approve once. `claude mcp list` shows both as connected.
- **Cursor**: `mkdir .cursor && cp .mcp.json .cursor/mcp.json`.
- **Other MCP clients**: register the two commands from `.mcp.json` by hand, with the
  `GOAD_REF` and `CODESTYLE_REF` values it pins. On a client that speaks the `claude
  mcp add` syntax:

  ```bash
  claude mcp add goad -e GOAD_REF=<tag from .mcp.json> -- \
    sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py'

  claude mcp add codestyle -e CODESTYLE_REF=<tag from .mcp.json> -- \
    sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/codestyle/$CODESTYLE_REF/codestyle_mcp.py'
  ```

If a server fails to connect, run its line from `.mcp.json` by hand in a terminal; the
traceback in the first lines is the error. The first start builds an environment and is
slow; that is not a failure.

Optionally, the course site https://learn.pttrn.io has a student MCP server that gives
your assistant the lessons, goals, rubric and your own feedback. Connect it at
https://learn.pttrn.io/link. The command contains a personal token: treat it like a
password.

## 6. Your own chat

Lesson 1 works on your own WhatsApp export.

1. Copy `config.example.toml` to `config.toml`. It is gitignored.
2. Export a chat from WhatsApp, put it in `data/raw/`, and name it `_chat.txt` (or
   change the name in `config.toml`).
3. The preprocessor parses timestamps with the `datetime_format` in `config.toml`.
   Check it against your export; the codes are in the
   [datetime documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).
4. Run it:

   ```bash
   uv run analyzer --device ios
   ```

   Use `android` for an Android export. This runs `wa_analyzer.preprocess:main` and
   writes a csv to `data/processed/`; the `logs/` folder holds a logfile with more
   detail. Expect output like:

   ```
   16:07:19.191 | INFO     | __main__:main:71 - Using iOS regexes
   16:07:19.201 | INFO     | __main__:process:61 - Found 1779 records
   16:07:19.202 | INFO     | __main__:save:30 - Writing to data/processed/whatsapp-20240211-160719.csv
   16:07:19.206 | SUCCESS  | __main__:save:32 - Done!
   ```

5. Put that csv name under `inputpath` in `config.toml`, run
   `notebooks/lesson1/01.3-your-own-chat.ipynb`, and put the `.parq` it writes under
   `current`. Every later notebook reads your chat through `load_own_chat()` from that
   `current` key, so switching chats is one line in `config.toml`.

To use the environment from a plain shell instead of through `uv run`:

```bash
source .venv/bin/activate
which python      # must be inside .venv
```

## 7. The graph dashboard

Once you have a processed file:

```bash
uv run streamlit run streamlit_app.py
```

## Without uv

If a managed laptop forbids installing uv, a plain Python 3.12 with pip still works:
`python -m venv .venv`, activate it, `pip install -e .`. The
[codestyle notes on dependency management](https://github.com/raoulg/codestyle/blob/main/docs/dependencies_management.md)
explain virtual environments and `pyproject.toml` if these are new. Expect to translate
every `uv run x` in the course material to `x` inside the activated environment.
