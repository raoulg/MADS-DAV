This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation". Eight lessons on your own chat data: cleaning and pipelines, comparing categories, time, distributions, relationships, vector spaces and social graphs, with `goad_toolkit` and `codestyle` as coaches.

# Setup

The setup is meant to be done together with an assistant. Hand it one of these URLs and ask it to walk you through; it takes one step at a time, explains why each tool is there, and keeps track of where you are between sessions.

## VM
The Linux lab VM, if you received an invite for one (do this first, then the tooling):

```
https://raw.githubusercontent.com/raoulg/MADS-DAV/main/references/lab-setup.md
```

## local
If you dont have a VM in this course, you can continu with the local setup: tools on your own machine (git, uv, VS Code, the pre-commit hooks, the coaches):

```
https://raw.githubusercontent.com/raoulg/MADS-DAV/main/references/tooling-setup.md
```

## manual setup

If you have no assistant with MCP access, or want every command on one page, [docs/setup.md](docs/setup.md) is the manual version: uv, the repository and the branch rule, the editor, the hooks, the coaches, and your own chat export.

Two rules that matter from day one, whichever route you take:

- **Work on your own branch, never on `main`**, and copy a notebook under a new name before editing it. `main` receives new material every week; a renamed file never conflicts with it.
- **Python and packages come from uv only.** No pip, no conda, no separately installed Python.
