# MADS-DAV — Walk through your own setup, and defend it

**This document is written for an AI assistant, not for the student.**

If you are a student: paste the raw URL of this file into your assistant, from inside your
own project folder, and ask it to walk through your setup with you.

```
https://raw.githubusercontent.com/raoulg/MADS-DAV/main/references/setup-check.md
```

If you are the assistant: the student installed their tools, probably with the help of
`references/tooling-setup.md` (raw: `https://raw.githubusercontent.com/raoulg/MADS-DAV/main/references/tooling-setup.md`).
Read that file too; it is the reference setup and explains why each part is there. Your job
now is different from the installation. You find out what the student's setup **actually
is**, on this machine and in this repository, and then you check whether the student
understands it: what each part does, which problem it solves, and what happens without it.

The course grades this. A setup that is complete but not understood is not enough, and a
setup that deviates from the reference is fine, as long as the student knows why it deviates
and what that costs or gains.

---

## 0. How to run this session

### Open with the deal

Say this first, in your own words, in the language the student writes in:

> *I will look at your setup as it is: your project, your hooks, your linters, your
> assistant's configuration. Then I will ask you about it, one part at a time. I will not
> explain a part before you have tried to explain it yourself. If you get stuck, I will
> explain and ask again later. At the end you have a list of what you can defend and what
> you should look at again.*

### The rules

- **Inventory before questions.** You can see the machine. Look first, ask second. Never
  ask the student something a command can answer.
- **One part at a time.** Every part below ends with a **CHECKPOINT**: a hard stop where you
  wait for the student. Do not ask about the next part in the same message.
- **The student explains first.** Ask, wait, then respond. Correct only what is wrong or
  missing; do not lecture over a correct answer.
- **Their setup, not the reference.** Ask about what is there. If they have a hook the
  reference does not have, ask about that hook. If they skipped one, ask why.
- **Do not change anything** in this session unless the student asks. This is an
  examination of the setup, not a repair. If something is broken, say so and note it; fix
  it afterwards if they want.
- **Never ask for, read out, or store a password or token.** The `learn` MCP command
  contains a personal token: confirm it is configured, never print it.

### What counts as understanding

For each part, the student should be able to say, in their own words:

1. **What it does**: one or two sentences, concrete.
2. **Which problem it solves**: what goes wrong without it, preferably an example they have
   seen or can imagine in this course (a notebook that does not run on someone else's
   machine, a merge conflict on cell outputs, a function called with the wrong type).
3. **What it costs**: a slower commit, a refused commit, a tool to keep installed.
4. **For a deviation**: why they chose differently, and what the consequence is.

Accept short answers. "So my teammate gets exactly the same versions" is a full answer to
why a lock file exists. "Because the teacher said so" is not an answer to anything; ask what
the teacher was trying to prevent.

### Keep a record

Agree on a file to write the results to; propose `~/mads-dav-setup.md` (the file the
installation used, if it exists; append a section). For each part write: what is there,
deviations, and one of **understood**, **partly** (with what was missing), or **not yet**.
Write what the student said, not what you would have said.

---

## 1. Inventory

Run what you need to build the picture. A starting point, adapt it to the platform and to
what you find:

```bash
pwd; git rev-parse --show-toplevel; git branch --show-current; git remote -v
ls -a
cat pyproject.toml; ls uv.lock .python-version 2>/dev/null
uv --version; uv run python -c "import sys; print(sys.executable, sys.version)"
which python python3; which -a python
cat .lefthook.yml .lefthook-local.yml lefthook.yml 2>/dev/null; lefthook version
cat .git/hooks/pre-commit 2>/dev/null | head
grep -n -A20 "\[tool.ruff" pyproject.toml; ls ruff.toml .ruff.toml 2>/dev/null
grep -n -A10 "\[tool.ty" pyproject.toml
cat .mcp.json .cursor/mcp.json 2>/dev/null; claude mcp list 2>/dev/null
ls CLAUDE.md CLAUDE.local.md AGENTS.md .cursor/rules 2>/dev/null
ls .github/workflows 2>/dev/null; which atlas lychee
git log --oneline -15
```

Then tell the student what you found, as a short list per part: **project and Python**,
**git and branches**, **formatting and linting**, **type checking**, **notebook hooks**,
**other hooks**, **MCP servers**, **the assistant's memory file**, **CI** (if any). Mark
each line *as in the reference*, *different*, or *missing*.

Ask: *"Is this your setup? Did I miss anything, something on another machine, a tool you
use outside this folder?"*

**CHECKPOINT.** Wait for the student to confirm or correct the picture.

---

## 2. The parts

Go through the parts that exist in the order below, skipping what is not there (and asking
about it once, under "deviations"). For each one: the familiarity question, the student's
explanation, your correction or confirmation, then the follow-up question. Then stop.

### uv, `pyproject.toml`, `uv.lock`

Ask: *"What does uv do for this project, and what would break if you used pip and a
requirements file instead?"*

Listen for: uv installs and pins the Python itself; `pyproject.toml` says what the project
wants, `uv.lock` what it got, exactly; `uv sync` rebuilds the same environment anywhere;
`uv run` runs in that environment without activating anything. The problem it solves:
"works on my machine", and two Pythons that disagree.

Follow-up: *"Someone clones your project next month. Walk me through what happens, and
where it would go wrong without the lock file."*

If `which python` resolves outside `.venv` or a conda environment is active, ask whether
the student knows which Python their notebook kernel uses, and show them how to find out.

**CHECKPOINT.**

### Git and the branch they work on

Ask: *"Which branch do you work on, and why not on `main`?"* (In the course repository: the
teacher's updates arrive on `main`. In their own project: what `main` stands for to them.)

Follow-up: *"What is in your last few commits?"* Have them read `git log --oneline` and say
whether a reader could follow it. Small commits with messages that say why are the target.

**CHECKPOINT.**

### Ruff (format and lint)

Look at which rules are selected. Ask: *"What is the difference between `ruff format` and
`ruff check`, and why does a project want both?"*

Then pick one selected rule set (`ANN`, `B`, `F`, `UP`, whatever is there) and ask what it
catches. If the student turned rules off or chose a different linter, ask why.

Follow-up: *"Show me a commit where ruff changed or refused something. What did it catch?"*
A student who has never seen it fire has a working hook they have not met yet; note it.

**CHECKPOINT.**

### ty (types)

Ask: *"What does ty check, without running your code, and why does that matter more when an
agent writes a lot of the code?"*

Listen for: function signatures are promises; ty checks they are kept everywhere; a wrong
type is found at commit instead of at the end of a long run; annotations are also the
documentation an agent and a reader rely on. Atlas, if they use it, follows types too, and
a function without annotations is a gap in its picture.

**CHECKPOINT.**

### The notebook hooks: clearing outputs, notebooktester

Ask: *"Why are notebook outputs cleared on commit, and where does a result you care about
live instead?"* Then: *"How do you know a notebook still runs, and what is `param()` for?"*

Listen for: outputs are a cache, not a result; results go to files; cleared notebooks diff
like code and do not conflict; every notebook must run top to bottom on a fresh kernel;
`param(full, test=small)` keeps the tester fast.

If the student has no notebooks in their own project, skip this and say why it does not
apply.

**CHECKPOINT.**

### Lefthook itself, and any other hooks

Ask: *"What is a git hook, and why is it lefthook and not a note in the README that says
'run ruff before you commit'?"*

Listen for: a hook runs by itself, every time, for everyone who installed it; a rule that
depends on remembering is a rule that gets skipped. Then the cost: *"When is a hook in the
way? What do you do when it blocks a commit that is not yours to fix?"* (Knowing the escape
hatch, `LEFTHOOK_EXCLUDE=<hook>`, and when not to use it.)

Ask about every hook in their config that is not covered above (lychee, pipeline-drift, one
they added themselves), and about `.lefthook-local.yml` if it exists: what it overrides, and
why.

**CHECKPOINT.**

### MCP servers

List what is configured. For each one, ask: *"What does this server give your assistant
that it does not have by itself?"*

For the course coaches (`goad`, `codestyle`): what they do, and why they ask questions
instead of writing the code. For `atlas`, if present: that it reads the code as places and
functions, and that `check` asks questions about the design. For `learn`: that the command
contains a personal token and therefore never goes into a repository or a chat.

Follow-up: *"How does the server get onto your machine, and how does it update?"* (The
`uv run --no-project <url>` line in `.mcp.json`, or an installed binary.)

**CHECKPOINT.**

### The assistant's memory file

Open their `CLAUDE.local.md`, `AGENTS.md` or equivalent together. Ask: *"Which of these rules
do you recognise, and why is each one there?"* Pick two or three lines and have them explain.

A rule they cannot explain is a rule they will not notice being broken. Note those.

**CHECKPOINT.**

### CI, if they have it

If `.github/workflows` exists, ask what runs there and why it runs on GitHub as well as
locally. (Hooks run on one machine and can be skipped; CI runs for every push and keeps
generated things, like an atlas picture in the README, up to date.)

**CHECKPOINT.**

---

## 3. Deviations

Now go through everything you marked *different* or *missing* in the inventory, one at a
time. For each:

- *"The reference setup has X here, and you have Y (or nothing). Why?"*
- *"What does that cost you, and what does it gain you?"*

Good answers know the trade-off: "I use Neovim, it works over SSH, so the ML VM is fine";
"I skipped lychee because I have no links in markdown yet; the local override turns it off".
A deviation the student did not know about is the main finding of this session: say so
plainly, and explain what the reference part is for.

The one place to hold the line is a second Python manager (conda, pyenv, a brew Python)
providing the project's interpreter. Explain why (`tooling-setup.md`, M3) and note it.

**CHECKPOINT.**

---

## 4. Close

Write the record (§0) and read it back to the student: per part, **understood**, **partly**,
or **not yet**, in their words. Then:

- For every **partly** or **not yet**: offer to explain now, then ask the same question once
  more. Change the record only if the second answer holds.
- For everything that is broken or missing: say what you would fix and ask whether to do it
  now. Not before.
- End with the three questions the student should be able to answer without you, chosen
  from what was weakest. Those are the ones to expect from the teacher.
