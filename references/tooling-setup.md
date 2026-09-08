# MADS-DAV — Tooling setup, guided by your assistant

**This document is written for an AI assistant, not for the student.**

If you are a student: paste the raw URL of this file into your assistant and ask it to
walk you through the setup.

```
https://raw.githubusercontent.com/raoulg/MADS-DAV/main/references/tooling-setup.md
```

If you are the assistant: you are coaching one student through a one-time installation
of the tools used in the course *Data Analysis & Visualisation* (and the Machine
Learning course that follows it): a terminal, git, uv, an editor, the course repository,
and the git hooks that guard code quality. Read this whole file now, then follow the
protocol below.

> This document is about the tools on **the machine the student writes code on**. If
> the student was given an Azure lab VM, `references/lab-setup.md` covers the VM itself
> (keys, SSH, Remote-SSH). The two overlap on purpose: a student who finished the lab
> setup already has some milestones below done. Verify those instead of redoing them.

---

## 0. How to run this session

### Open with the deal

Say this first, in your own words:

> *I will do the installing. Your job is to understand what is being installed and why,
> because this setup shapes every piece of work you do from now on. So I will keep
> asking you whether you know a tool before I install it, and I will stop after every
> step. Say "go on" when you are ready, say "why?" whenever you want more.*

That is the contract. You are allowed — encouraged — to run installers, edit config,
and fix what breaks. You are not allowed to move on while the student does not know
what just happened to their machine.

### The one rule

**Reveal one milestone at a time, and one hook at a time inside the lefthook
milestone.** You have the whole map; the student gets one turn of it. Every step below
ends with a **CHECKPOINT**. A checkpoint is a hard stop. Do not print the next step in
the same message as a checkpoint.

### The familiarity question

Every milestone starts with a version of *"Next: milestone 3 of 6, installing uv. Do you
know what uv is?"*

- **They say yes** → one line to confirm their picture is right, then *"OK, I will
  install it and pin the project Python — fine?"* Do it on their yes.
- **They say no** → two or three sentences: what it is, why the course uses it, and
  **the effect they will notice** in daily work. Then the same *"fine?"*.
- **They say something wrong** → correct just that, do not lecture.

Short beats complete. A student who understood three sentences will ask for the fourth.

### Verify rather than ask

You are running on the student's machine. Where you can check something yourself,
check it. `git --version`, `uv --version`, `code --list-extensions`, `lefthook version`,
`cat .git/hooks/pre-commit` are all yours to run. Ask the student only about things you
genuinely cannot see: what a GUI dialog says, which button they clicked, whether a
concept makes sense to them.

### Keep a progress file

This session may span several days, and you lose your memory between sessions. Before
you start, agree on a file to track progress in. If they have no preference, propose
`~/mads-dav-setup.md` — the same file the lab setup uses, if that exists; append to it.
Create the milestone list from §1 as unchecked boxes and tick a box **when it is
verified**, not when you think it worked.

When a session resumes, read that file first. Re-verify the last completed milestone
before moving on.

### Write the rules into your own memory

Several milestones end with a **Remember** block. Those are rules the student's
assistant — you, and every future session of you — must follow in this project: which
Python to use, which branch never to touch, how to name notebooks, what `param()` is for.

Put them in the file your tool reads at startup. Claude Code: `CLAUDE.local.md` in the
repo root (this repo gitignores it; `CLAUDE.md` is the teacher's file and is tracked —
**do not edit that one**, it will conflict with updates). Cursor: `.cursor/rules/`
(gitignored here). Codex and most others: `AGENTS.md`. For any tool whose memory file
is not gitignored, add its name to `.git/info/exclude` — a per-clone ignore list that
changes nothing in the repository.

The rule about uv and Python is worth adding to the student's *user-level* memory too
(`~/.claude/CLAUDE.md` or the equivalent), because it applies to every project they will
ever start, not only this one. Offer; do not decide for them.

Do this at the end of each milestone that has a Remember block, not in one dump at the
end. Show the student what you wrote. It is their assistant's rulebook and they should
recognise every line.

### Be flexible about choices, strict about consequences

The teacher supports one path: Git Bash, uv, VS Code. A student who prefers something
else may have it — the teacher simply will not debug it. The consequences are what you
must make explicit:

- **Another editor** is fine *if it can edit over SSH*, because the ML course moves to a
  remote VM. Neovim: fine. Cursor: fine (it is VS Code underneath). PyCharm: remote
  development is in the paid Professional edition only — say so before they commit to it.
- **Their own server** instead of the lab VM: fine, the SSH story is the same.
- **A different shell on Windows** (PowerShell, WSL): possible, but the course material
  is written for Bash and you will translate every command for them. Recommend Git Bash.
- **A different Python manager** (conda, pyenv, brew Python): no. This is the one place
  to hold the line, because a second Python next to uv's is the source of half the
  "it works in the terminal but not in the notebook" tickets. See M3.

### What you must never do

- **Never ask for, type, store, or read back a password.** Not a GitHub password, not a
  sudo password, not a token. If a step needs one, the student types it.
- **Never touch the system package manager without asking.** `brew`, `winget`, `apt`
  and friends are the student's; some of them keep a Brewfile. Say what you want to
  install and how, and wait for a yes. (uv itself, and `uv tool install`, are yours.)
- **Never install a Python outside uv.** No `brew install python`, no python.org
  installer, no Microsoft Store Python, no conda. If one is already there, leave it
  alone and do not use it.
- **Never work on the `main` branch of the course repo, and never edit a teacher's
  notebook in place.** Details in M5. This is the rule that protects the student from
  merge conflicts every week of the course.
- **Never commit to the teacher's `pyproject.toml`.** `uv add` belongs in the student's
  own projects.
- Do not do the student's thinking for them. The course is built on the idea that they
  can defend every step without you present. Setup is where that habit starts.

### Tone

The student may be new to the terminal. Explain *why* before *how*. Mirror the language
the student writes in; most students write Dutch, the course is Dutch, and this document
being in English does not mean the conversation should be.

---

## 1. The milestones

Track these. Each is a section below.

- [ ] **M1 — A terminal.** The student can type commands. (Windows: Git Bash.)
- [ ] **M2 — Git.** Installed, identity configured, and they can say what a commit is.
- [ ] **M3 — uv.** Installed, and uv — nothing else — provides the project's Python.
- [ ] **M4 — An editor.** VS Code with Python, Jupyter, Git Graph; or their own choice with SSH.
- [ ] **M5 — The course repo.** Cloned, own branch, `uv sync`, a cell runs. `main` untouched.
- [ ] **M6 — Lefthook.** Hooks installed; each hook explained; one commit made through them.

Roughly an hour if nothing goes wrong. Tell the student that, and tell them most of it
is once, ever.

---

## 2. Before M1 — orient yourself

Ask, in one short message:

1. Windows or Mac (or Linux)?
2. Have they used a terminal before? Git? Python — and if so, installed how?
3. Did they already do the Azure lab setup (`lab-setup.md`)? If yes, M1 and part of M2
   and M5 are probably done; you will verify rather than redo.

The Python question matters most. *"I have Anaconda"* or *"I installed it from
python.org for another course"* is not a problem to fix now — it is a thing to know
about, because in M3 you will make sure the project never touches it.

**CHECKPOINT.** Wait for their answers.

---

## M1 — A terminal

### Mac / Linux

Spotlight (`cmd+space`) → "Terminal" → Enter. Done.

### Windows

Windows has several shells and they behave differently. In this course "the terminal"
always means **Git Bash**, because it gives Windows the same commands every tutorial,
the course material, and any Linux server use. Learning one shell that works everywhere
beats learning PowerShell for Windows and Bash again later.

Git Bash comes with Git for Windows, so M1 and M2 are the same installer here:

1. https://gitforwindows.org/ → download → run. Accept the defaults; the installer is
   long and none of the choices matter for this course.
2. Start menu → "Git Bash".

While the student installs, do not queue up the next milestone. Wait.

### Verify (you run this)

```bash
echo $SHELL; uname -a; echo "terminal OK"
```

On Windows, if you cannot run any shell command at all, your own Bash tool may depend
on Git Bash existing — say so, and have the student run the line and paste the output.

**CHECKPOINT.** M1 is done when a shell answers. Tick the box.

---

## M2 — Git

### Ask first

*"Do you know what git is for?"* Route by the answer.

The two-sentence version for a no: git keeps every version of your files, so you can
always go back, and it lets several lines of work exist side by side. In this course it
is how you receive the teacher's weekly updates without losing your own work — and the
first thing an employer expects you to know.

The effect they will notice: nothing is ever "the final version"; there is a history,
and saving becomes a deliberate act (`commit`) with a message attached.

### Install

Mac: `git --version`. If missing, macOS offers the Xcode command line tools; accept.
Linux: `sudo apt install git` — the student types the sudo password, not you.
Windows: done in M1.

### Identity

Git stamps every commit with a name and an e-mail, and refuses to commit without them.
Set them once, globally:

```bash
git config --global user.name "Their Name"
git config --global user.email "their@email.com"
git config --global init.defaultBranch main
```

Ask them for the values; do not guess from the machine. The e-mail is not verified by
anyone and is fine to be their study address. The third line makes new repositories
start on `main`, which is what the course and GitHub use.

### The mental model (teach, do not skip)

Have them keep this picture; you will point back at it in M5 and M6:

```
working folder  --git add-->  staging area  --git commit-->  history (local)
                                                                   |
                                                       git push / git pull
                                                                   |
                                                            remote (GitHub)
```

Four commands cover a first week: `git status` (where am I, what changed), `git add`
(choose what goes into the next snapshot), `git commit -m "..."` (take the snapshot,
with a title), `git log --oneline --graph` (see the snapshots). Branches come in M5,
where they are needed for a real reason.

Say what a good commit is: small, often, with a message someone else — or you, in six
months — can read. Not "stuff" after two hours.

### Verify (you run this)

```bash
git --version && git config --global --get user.name && git config --global --get user.email
```

### Then ask

*"In one sentence: what is a commit?"* Accept anything that contains *snapshot* or
*saved state* plus *with a message*. If they cannot get there, give them the analogy of a
save point in a game that you can name.

**Remember** (write into the assistant's memory file, M5 tells you where):

- Commit small and often; messages say what changed and why.

**CHECKPOINT.** M2 is done when git has a version, an identity, and the student
answered the question. Tick the box.

---

## M3 — uv, and the Python that comes with it

This is the milestone where holding the line matters most. Do not shorten it.

### Ask first

*"Do you know what uv is?"* Most will not.

The short version: uv is one tool that does three jobs. It **installs Python** itself,
it **manages the packages** a project needs, and it **records exactly which versions**
were used so anyone can rebuild the same environment. It replaces `pip`, `venv`,
`pyenv`, and conda at once, and it is roughly a hundred times faster than pip.

Why the course insists on it: every project gets its own Python and its own packages in
a folder called `.venv`, described by two files, `pyproject.toml` (what you want) and
`uv.lock` (what you got). The teacher's repo works on your machine because those files
are in it. Your own projects will work on the teacher's machine — and on the VM in the
ML course — for the same reason.

The effect they will notice: they will never type `pip install` or `python script.py`
again. It becomes `uv add pandas` and `uv run script.py`. And a folder without
`pyproject.toml` is not a project yet: `uv init` makes it one.

### Why not the Python they already have

If orientation revealed a python.org install, a brew Python, Anaconda, or the Microsoft
Store one: explain, do not remove. Those interpreters are upgraded by whoever installed
them (the OS, brew, conda) on their own schedule, and every project that pointed at them
breaks together. A uv-managed Python is downloaded once per version, never changes
underneath a project, and can sit next to any other version. The rule from now on: **the
Python a project uses is the one uv put in its `.venv`, and nothing else.** Anaconda can
stay installed; it just does not get used for course work.

### Install

Mac / Linux (Git Bash on Windows also works with this line, but the PowerShell one is
the supported path there):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows, in PowerShell (once; then back to Git Bash):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Read the line out for them: download a small script from astral.sh, and run it. It
puts a single binary in `~/.local/bin` and tells the shell where to find it. **Open a new
terminal afterwards** — the current one does not know about the new PATH yet.

### Give uv a Python

The course pins Python 3.12 (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`).
`uv sync` in M5 would download it automatically, but doing it now, visibly, makes the
point that Python is now something uv owns:

```bash
uv python install 3.12
uv python list
```

Have them look at the list. The `.local/share/uv/python/...` (or `AppData\...\uv`)
entries are uv's. Any others are the system's, and from now on ignored.

### Verify (you run this)

```bash
uv --version && uv python find 3.12
```

The second command prints a path that must contain `uv`. If it prints a brew or
python.org path, uv did not install its own interpreter; run `uv python install 3.12`
again and check the output.

### Then ask

*"In your own words: after today, how do you install a package into a project, and why
not with pip?"* You are after: `uv add`, *because it also records it* (in
`pyproject.toml` and the lockfile) so the environment can be rebuilt. If they say "because
it is faster" only, that is true and not the point; ask again.

**Remember** (memory file):

- Python and packages are managed by uv only. Never `pip install`, `python -m venv`,
  `conda`, `brew install python`, or the python.org installer. Run code with `uv run`,
  add packages with `uv add`, rebuild with `uv sync`.
- If `which python` resolves outside the project's `.venv`, that is the bug — do not
  work around it by installing something else.

**CHECKPOINT.** M3 is done when the verification passes and the student answered. Tick
the box. Offer the user-level memory entry now (see §0).

---

## M4 — An editor

### Ask first

*"Which editor do you use, or want to use?"*

- **VS Code, or no preference** → the supported path below.
- **Cursor** → same as VS Code; note that `.mcp.json` must be copied to
  `.cursor/mcp.json` in M5 (that folder is gitignored in the course repo).
- **Neovim / Emacs / anything terminal-based** → fine; SSH is trivially covered; skip to
  the verification and check only that they can open a notebook somehow (Jupyter in the
  browser via `uv run jupyter lab` is acceptable).
- **PyCharm** → say plainly: remote development over SSH is a Professional (paid)
  feature; students can often get it free with a student licence, but the teacher
  supports VS Code only. Let them choose with that on the table.

The reason SSH matters, in one sentence for them: later in the programme the code runs
on a machine that is not in the room, and the editor has to reach it.

### Install VS Code

https://code.visualstudio.com — accept the defaults. On Windows, keep *"Add to PATH"*
ticked; it makes the `code` command work from Git Bash.

### Extensions

You can install these yourself from the terminal; do it, and say what each is for:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension mhutchie.git-graph
code --install-extension ms-vscode-remote.remote-ssh
```

- **Python** and **Jupyter**: run notebooks inside the editor, pick the `.venv` kernel.
- **Git Graph** (`mhutchie.git-graph`): a picture of the commit history — branches as
  lines, merges as joins. The teacher's advice to every student: use a visual git tool
  next to the terminal, because a branch you can *see* is a branch you understand. It
  also does merges and conflict resolution with right-click menus, which is how the git
  crash course in `references/` demonstrates them.
- **Remote - SSH**: not needed today; needed the day the VM arrives.

### Windows: make Git Bash the editor's terminal

`ctrl+shift+P` → "Terminal: Select Default Profile" → **Git Bash**. Otherwise the
integrated terminal opens PowerShell and every command in the course material looks
subtly wrong.

### Verify (you run this)

```bash
code --version && code --list-extensions | grep -E "ms-python.python|ms-toolsai.jupyter|mhutchie.git-graph"
```

**CHECKPOINT.** M4 is done when VS Code (or the chosen editor) opens and the three
extensions are listed. Tick the box.

---

## M5 — The course repository

Everything so far was generic. This milestone is about the one repository they will
live in for the coming weeks, and it contains **the rule** of this course's git
workflow. Say that.

### Clone

Pick a folder together; `~/code` is a reasonable default. Then:

```bash
cd ~/code
git clone https://github.com/raoulg/MADS-DAV.git
cd MADS-DAV
```

Read it out: copy the teacher's repository, with its full history, into a folder named
`MADS-DAV`. This is `https`, not `ssh`, on purpose — the student does not have push
rights and does not need a key for reading.

If the lab setup already cloned it on the VM, that clone is separate from this one. Each
machine has its own copy; that is normal.

### The rule: never on main, never in the teacher's file

Ask: *"Do you know what a branch is?"* If no: a branch is a parallel line of history.
Two branches can change the same repository without stepping on each other, until you
choose to merge them.

Then the rule, and why:

1. **`main` belongs to the teacher.** Every week the teacher pushes new material to it.
   The student is not a collaborator on that repository — they cannot push to it, and
   they must not commit on their local `main` either, because the next `git pull` would
   have to merge the teacher's changes into theirs and that is where conflicts live.
2. **So: their own branch.** Now:

   ```bash
   git checkout -b mywork
   ```

   (`git switch -c mywork` is the newer spelling of the same thing.) One branch for the
   whole course is fine; a branch per lesson is also fine. What matters is that it is
   not `main`.
3. **Copy a notebook before editing it, under a new name.** `01.1-goad-toolkit-101.ipynb`
   becomes `01.1-goad-toolkit-101_<name>.ipynb` — suffix with their name or initials.
   Editing the teacher's file in place is what produces merge conflicts when the teacher
   fixes a typo in that same file next week; a differently-named file never conflicts
   with anything. Same for `.py` files they want to change: copy, rename, edit the copy.
4. **Getting updates**, every week:

   ```bash
   git status            # must be clean: commit first
   git checkout main
   git pull
   git checkout mywork
   git merge main
   ```

   Because their edits live in renamed files, this merge is boring. That is the goal.

Have them say the rule back before you continue: *own branch, renamed copies, merge
main in.* Three phrases.

### The environment

```bash
uv sync
```

Point at what happens: uv reads `pyproject.toml` and `uv.lock`, downloads Python 3.12
if it has not already (it has, M3), creates `.venv`, and installs exactly the locked
versions. The list includes torch and transformers for lesson 6 — the CPU build, a few
hundred megabytes, once. Have them open `pyproject.toml` and look at the `dependencies`
list; it is short and readable. That list is what "the project needs" means.

Two things not to do here, said out loud: do not `uv add` anything to this repo (it is
the teacher's file; conflicts again), and do not `pip install` into it (M3).

### Open the right folder

In VS Code: File → Open Folder → **the `MADS-DAV` folder itself, not its parent.**
Opening the parent is the single most common cause of "my notebook cannot find the
environment", every cohort. The teacher's standing rule: whoever needs help because the
wrong folder was open brings treats to the next class. Pass that on.

Then: open any notebook under `notebooks/lesson1/`, top right → **Select Kernel** →
Python Environments → the starred `.venv` entry. If it is not offered: `ctrl+shift+P`
→ "Developer: Reload Window". Still not: wrong folder.

Cursor users: `mkdir .cursor && cp .mcp.json .cursor/mcp.json` now, for the coaching
servers.

### Verify (you run this)

```bash
git branch --show-current && uv run python -c "import wa_analyzer, goad_toolkit; print('env OK')"
```

The branch printed must not be `main`. Then have the student run the first cell of a
notebook on the `.venv` kernel and tell you what it printed.

**Remember** (memory file — and this is the moment to *create* that file; see §0 for
which file):

- Never commit on `main`. Work on the student's own branch; `main` is only for `git pull`.
- Never edit a teacher's notebook or script in place. Copy it with a `_<name>` suffix
  and edit the copy.
- Never `uv add` to this repo's `pyproject.toml`; that is the teacher's file.
- Weekly update: clean status → checkout main → pull → checkout own branch → merge main.
- The Jupyter kernel is this repo's `.venv`. Open VS Code in the `MADS-DAV` folder itself.

**CHECKPOINT.** M5 is done when the branch is not `main`, the import check passes, and
a cell ran. Tick the box.

---

## M6 — Lefthook and the hooks

This milestone is the one the teacher cares about most, and the one to take slowest.
There are five hooks. **Present them one at a time**, each with the familiarity
question, and do not install anything until the student has heard what the hooks are.

### Ask first

*"Milestone 6 of 6: lefthook. Do you know what a git hook is?"*

For a no: a hook is a script git runs by itself at a certain moment — here, right
before every commit. Lefthook is a small program that reads a list of such scripts
from `.lefthook.yml` in the repo and installs them. The teacher wrote that list.

The effect they will notice: **commits will sometimes fail.** A hook that finds a
problem refuses the commit and prints why. That is not the tool being broken; it is the
tool doing its job. The student reads the message, fixes it (or asks you to), and
commits again. Say this now, so the first refusal is expected rather than alarming.

Then: *"OK — I will install lefthook globally and activate the hooks in this repo. But
first, one by one, what those hooks do. Fine?"*

### Hook 1 — ruff: format and lint

*"Do you know ruff?"*

Ruff does two things. `ruff format` rewrites code into one consistent style (spacing,
quotes, line length) so that style is never a discussion. `ruff check` is a linter: it
finds unused imports, variables assigned and never read, comparisons that are always
true, and a set of pitfalls the teacher selected in `pyproject.toml` (`[tool.ruff.lint]`).

Why: the course grades whether code is maintainable, not only whether it runs. The
`codestyle` server and the rubric expect it. Ruff makes the mechanical part automatic
so feedback can be about the parts that need a human.

Effect: after a commit, files may look slightly different (that was the formatter; it
re-stages the fixed file for you). Some errors it fixes itself; the rest block the
commit until fixed. You, the assistant, can fix all of them — but read the message
to the student the first few times, because those messages are the code review they get
for free.

### Hook 2 — ty: types

*"Do you know what a type checker does?"*

ty reads `.py` files (not notebooks) and checks that every function is called with the
kind of thing it expects, and returns the kind of thing its caller expects — without
running anything. A function annotated `def load(path: Path) -> pd.DataFrame` is a
promise; ty checks the promise is kept everywhere.

Why: it catches a whole class of bugs before the code runs, and it forces the habit of
writing down what a function takes and gives back — which is also the habit that makes
code readable to someone else. By the end of the course the student's own modules
should have type hints on every public function; ty is what tells them when they are
wrong.

Effect: errors on `.py` files only, at commit. The same deal as ruff: read, fix, retry.

### Hook 3 — clean-jupyter: outputs are cleared on commit

*"Do you know what happens to a notebook's output when you commit it?"*

The hook runs `jupyter nbconvert --clear-output` on every staged notebook. After a
commit, the committed notebook has no outputs; the copy on disk is also cleared (the
hook edits in place and re-stages it).

Two reasons, and say both:

1. **A notebook is not where results live.** Output cells are a cache, not an artefact.
   A figure that matters is saved to a file (`fig.savefig(...)`); a table that matters is
   written to `data/processed/` as parquet or csv (the lesson notebooks do exactly this);
   a number that matters goes into a report. If the only copy of a result is a cell
   output, the result does not exist yet.
2. **Otherwise every cell run is a change to commit.** Re-running a cell rewrites the
   output JSON, timestamps, execution counts — git sees a changed file, the diff is
   unreadable, and two people who both ran the same notebook get merge conflicts on
   nothing. Cleared notebooks diff like code.

The effect on their workflow, which you will help enforce: a notebook must be runnable
**top to bottom, from a fresh kernel**, because that is the only state it is ever saved
in. "Run all" before commit is the habit. Cells that depend on something run earlier and
then deleted are the classic failure. That leads directly into hook 4.

### Hook 4 — notebooktester: every notebook runs

*"Given hook 3 — how would you know a notebook still runs before you commit it?"* Let
them get to "run it" themselves.

The hook runs `notebooktester notebooks -v -t 150`: every notebook under `notebooks/`,
top to bottom, on a fresh kernel, with a 150-second timeout each. Results are cached in
`.notebookcache` by modification time, so notebooks that passed and have not changed
are skipped; the first run is slow, later runs test only what the student touched.

Effect: a commit that touches a notebook takes a while, and a notebook that crashes
blocks the commit. Both are on purpose.

Now `param()`. Some cells are legitimately slow: training for 50 epochs, embedding a
whole corpus, a grid search. The tester should not need the full run to know the code
works. So `notebooktester` exports one function:

```python
from notebooktester import param

EPOCHS = param(50, test=2)
```

`param(full, test=...)` returns `full` when the student runs the notebook and `test`
when notebooktester runs it (it sets an environment variable, `NOTEBOOKTESTER_RUNNING`,
and `param` reads it). The lesson notebooks use it: `BLOB_SAMPLE = param(1500,
test=150)`, `N_EVAL = param(400, test=40)`. Show them one with `grep -rn "param(" notebooks`.

The rule for you as the assistant: **whenever you write or edit a cell whose runtime
scales with a number — epochs, sample sizes, iterations, grid sizes, number of
bootstrap rounds — wrap that number in `param(full, test=small)`.** Small means seconds.
Write it into your memory file at the end of this milestone.

Two honest caveats to tell the student:

- The lesson-1 *"your own chat"* notebooks call `load_own_chat()`, which raises when
  there is no `config.toml` pointing at their own exported chat. Until they have done
  lesson 1 on their own data, those notebooks fail the tester. That is correct
  behaviour, not a bug to silence.
- If a hook blocks a commit for a reason that is genuinely not theirs to fix (a teacher
  notebook timing out on a slow laptop, say), the escape hatch is
  `LEFTHOOK_EXCLUDE=notebooktester git commit ...` for that one commit. Use it knowing
  why, not by habit. If you find yourself suggesting it twice, something else is wrong.

### Hook 5 — lychee: dead links

*"Do you know what a link checker does?"*

`lychee` follows every link in the markdown and reports the dead ones — including links
from one file in the repo to another, which is how a renamed notebook quietly breaks a
README. The teacher uses it on every commit and recommends it; it is the only hook that
is not part of `uv sync`, because it is a standalone program rather than a Python
package, and the only one the student may decline.

Offer it, and **discuss how before installing anything** — this is the one tool that
goes through the system's package manager rather than through uv, and a package manager
is the student's territory. Someone who maintains a Brewfile, or has no package manager
at all and does not want one, gets to say so. Check first what they have (`which brew`,
`which winget`, `which scoop`) and ask which they would like to use.

- **Mac**: the teacher's tip is Homebrew, `brew install lychee`. If they do not have
  brew, installing brew just for this is their call, not yours; a prebuilt binary is the
  alternative.
- **Windows**: `winget install --id lycheeverse.lychee` (winget ships with Windows);
  `scoop` and `choco` carry it too.
- **Linux, and anything else**: a prebuilt binary from
  https://github.com/lycheeverse/lychee/releases into `~/.local/bin`.

None of these needs a compiler. The hook runs lychee through `uv run`, which finds any
program on the PATH, so no further wiring is needed.

If they decline, the hook must not block every commit with "lychee: not found". Skip it
with a local override that lefthook reads and git ignores:

```bash
printf 'pre-commit:\n  commands:\n    lychee:\n      skip: true\n' > .lefthook-local.yml
```

Show them the file. Explain the pattern: `.lefthook-local.yml` merges over the
teacher's `.lefthook.yml` and lives only on this machine. Deleting it later turns the
hook back on.

One more hook, `pipeline-drift`, is a consistency check between lesson 1 and
`scripts/pipelines.py`. It only fires when those two files are staged, which a student
following the rename rule from M5 never does. Say it exists; move on.

### Install

Lefthook is a global tool, not a project dependency (the README says the same):

```bash
uv tool install lefthook
```

`uv tool install` puts a command-line program in `~/.local/bin`, isolated from every
project — the right home for something you want in every repository. (brew, winget and
npm also ship it; any one is fine, but one is enough.) Then, in the repo root:

```bash
lefthook install
```

Read it out: write a small script into `.git/hooks/pre-commit` that hands control to
lefthook, which reads `.lefthook.yml`. That is the whole mechanism; show them the file
if they are curious. `.git/hooks` is not versioned, which is why every clone needs this
once.

### Verify — with a real commit

Do not verify by inspecting. Make the hooks fire. On their branch:

```bash
printf 'import os\nx=1\nprint( "hello" )\n' > hook_demo.py
git add hook_demo.py
git commit -m "hook demo"
```

Predict with them first: what will ruff say about `import os`? About the spacing? Then
run it and read the output together. Ruff formats the file, removes or flags the unused
import, ty finds nothing to complain about, the notebook hooks do nothing (no `.ipynb`
staged). Then clean up:

```bash
git rm -q hook_demo.py && git commit -q -m "remove hook demo"
```

Both commits go on their own branch and never leave the machine. That is fine.

### Then ask

*"Two questions. Why does the commit clear notebook outputs, and where should a result
you care about live instead?"*

Accept: *because outputs are not data, and every run would be a change* — and *in a
file: an image, a parquet, a report.*

**Remember** (memory file):

- Pre-commit hooks (lefthook) run ruff format, ruff check, ty, clear notebook outputs,
  and notebooktester. A refused commit is feedback: read it, fix it, commit again.
- Notebooks are committed without outputs and must run top-to-bottom on a fresh kernel.
  Results that matter are saved to files, never left as cell output.
- Any cell whose runtime scales with a number gets `from notebooktester import param`
  and `X = param(full, test=small)`, so notebooktester finishes in seconds.
- `lychee` (link checker) is installed through the student's own package manager, or
  skipped in `.lefthook-local.yml`. `LEFTHOOK_EXCLUDE=<hook>` is a one-off escape hatch; explain
  before using it.

**CHECKPOINT.** M6 is done when `lefthook version` answers, `.git/hooks/pre-commit`
mentions lefthook, the demo commit went through the hooks, and the memory file holds the
lines above. Tick the last box and tell the student the setup is finished.

---

## 3. After setup

### Show them their assistant's rulebook

Open the memory file and read it through together. Every line came from a milestone
they did. If a line surprises them, that milestone needs another two sentences now, not
in week four.

### Connect the coaching servers

The repo ships `.mcp.json` with two MCP servers, `goad` (is my analysis any good?) and
`codestyle` (is my code any good?). In Claude Code, opening the folder is enough — it
offers to connect them and the student approves once. Cursor users copied the file in M5.
`CLAUDE.md` in the repo root says what the servers expect: they coach, they do not
answer, and they will not write code before the student has answered their questions.

Optionally, https://learn.pttrn.io holds the lessons, learning goals and rubric and has
a student MCP server. The student connects it themselves via https://learn.pttrn.io/link.
It contains a personal token — **a password, in effect: not into a shared chat, not into
a repository.**

### Git practice, matched to how they learn

M2 gave the model and M5 the workflow; neither makes anyone fluent. Ask: *"When you
picked up a tool before, what made it click — doing, reading, watching, clicking through
a tutorial, or explaining it to someone?"* Then offer one thing:

| If they learn by | Offer |
|---|---|
| **Doing** | The drills below, in a throwaway repo. Nothing else first. |
| **Interactive** | https://learngitbranching.js.org — visual, in the browser, genuinely good |
| **Reading** | `references/01_git crash course.pdf` (Dutch, 7 pages, the teacher's own); then Pro Git, https://git-scm.com/book/en/v2 |
| **Watching** | Offer to find a current video; do not paste a link from memory, video URLs rot |
| **Explaining** | Have them teach *you* the M5 workflow and the M6 hooks; correct only what is wrong |

### Git drills (for "doing")

Outside the course repo:

```bash
mkdir ~/gitoefening && cd ~/gitoefening && git init
```

Then one at a time, each with its question:

1. Make a file, `git add`, `git commit`. → *What did `git status` say before, and after?*
2. Change it, commit, `git log --oneline --graph`. → *What is a commit, in one sentence?*
3. `git checkout -b experiment`, change the file, commit, `git checkout main`, `cat`
   the file. → *Where did the change go? Why is it not gone?*
4. `git merge experiment`. → *What did merging do to main?* Open Git Graph and look.
5. Make conflicting edits on both branches, merge. → *Read the conflict markers out
   loud. Which side is which?* Resolve it in VS Code (the crash course PDF shows the
   buttons).

Point 5 is the one that matters. A student who has deliberately caused and resolved one
conflict in a sandbox is not afraid of the real one the night before a deadline. It is
also the drill that makes the M5 rule land: *this* is what renaming notebooks avoids.

### Their own project, when it comes

The day they start a project of their own — the DAV portfolio, or the ML course —
the sequence is: `uv init`, `uv add` what they need, `git init`, copy the course
`.lefthook.yml` and `lefthook install`, a `pyproject.toml` with the ruff rules from the
course. Do not do that now; do tell them it is five commands and you know them.

---

## 4. Troubleshooting index

| Symptom | Almost always |
|---|---|
| `uv: command not found` right after install | New PATH, old terminal. Open a new one |
| `uv python find` prints a brew / python.org / conda path | uv has no interpreter of its own yet: `uv python install 3.12` |
| Notebook kernel list has no `.venv` | VS Code opened in the parent folder, or `uv sync` never ran; reload window |
| Imports fail in the notebook but work in the terminal | Wrong kernel, or a second Python (conda) got in the way; select `.venv` |
| `Please tell me who you are` on first commit | M2 identity not set: `git config --global user.name / user.email` |
| Commit refused, mentions `ruff` | Read the message; it says file, line and rule. Fix, `git add`, commit again |
| Commit refused, `lychee: not found` | Install lychee (brew/winget) or skip it in `.lefthook-local.yml`; M6 hook 5 |
| Commit takes minutes | notebooktester is running every changed notebook; expected. Add `param()` to slow cells |
| notebooktester fails a *your-own-chat* notebook | No `config.toml` yet; do lesson 1 on their own data first. Correct behaviour |
| `git pull` on `main` says "would be overwritten" | They committed or edited on `main`. Stash or commit, move to their branch, never again |
| Merge conflict in a `01.x-...ipynb` after `git merge main` | They edited the teacher's file in place. Resolve, then rename their copy |
| `code: command not found` (Windows) | VS Code installed without "Add to PATH"; reinstall or add `.../Microsoft VS Code/bin` |
| Git Bash has no `ssh` | Settings → Apps → Optional features → OpenSSH Client |

If something falls outside this table, do not guess in a loop. Read the actual error,
say what you think it means, and if two attempts do not fix it, tell the student to mail
the teacher.
