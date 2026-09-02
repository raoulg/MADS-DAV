This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation", previously known as "Data Mining & Exploration". All instructions assume a UNIX machine (Linux or Os X). You should have received an invite link for a linux VM; if not, contact your teacher. On the VM, everything is installed (like uv).

The manual for setting up the VM can be found in the `references` folder, in addition to a `git_crash_course` to help you work with git. Make sure to read both documents before you start asking questions about either.


# Setup the virtual environment

## install with `uv` (recommended)
`uv` is the modern dependency manager for python, and it is quickly being adopted by the industry.
Besides being the fastest manager out there, is also has a very robust development team behind it.

1. Make sure you have `uv` installed.
You can check this by typing `which uv` in your bash terminal. If that doesnt return a location but `uv not found` you need to install it: On Unix systems, you can use `curl -LsSf https://astral.sh/uv/install.sh | sh`, for Windows read the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)
2. check if the `MADS-DAV` folder is already available cloned to the machine. If not, clone it yourself with the command `git clone https://github.com/raoulg/MADS-DAV.git`
3. Navigate to the MADS-DAV folder where the `pyproject.toml` is located with `cd MADS-DAV` and run `uv sync --all-extras`. This should create a virtual environment (a `.venv` folder) and install all dependencies there.
4. Read through the [uv docs](https://docs.astral.sh/uv/guides/projects/) "working on projects" intro into `uv`

## use the course assistant

This repo ships a `.mcp.json`. If you're on Claude Code, opening the folder is enough — it will offer to connect two coaching servers, `goad` and `codestyle`, and you approve them once. If you use Cursor, create a `.cursor` folder and copy `.mcp.json` into it as `.cursor/mcp.json`; Cursor reads the same format from there. See [CLAUDE.md](CLAUDE.md) for what the servers do and what's expected of you when you use them.

On any other MCP client, install both by hand:

```bash
claude mcp add goad -e GOAD_REF=v0.2.12 -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py'

claude mcp add codestyle -e CODESTYLE_REF=v0.2.0 -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/codestyle/$CODESTYLE_REF/codestyle_mcp.py'
```

## installation with pip (not recommended)
If for some reason you are unable to install `uv` (eg because you have a company laptop with restrictions on what to install) you can probably still install your `.venv` with base python and pip. Skip these steps if you already installed with `uv`
1. Open a bash terminal in the folder where you cloned the repo
2. create a `.venv` with `python -m venv .venv`
3. activate the `.venv` and run `pip install -e .`. If you are new to `.venv`s and `pyproject.toml` files, or dont know how to activate a `.venv`, you can read all the details in the [codestyle repo](https://github.com/raoulg/codestyle/blob/main/docs/dependencies_management.md)

# Run the preprocessor
Copy the `config.example.toml` file to a `config.toml` file. Update the contents after running the preprocessor.
Download a chat from Whatsapp and put it in the `data/raw` folder. Rename the file to `_chat.txt` (or change the `config.toml` file).

# NOTE
If you want to use my code in your own repo, do not copy paste everything. Instead, install it as a package; I published it on [pypi](https://pypi.org/project/wa-analyzer/) so you can simply do `uv add wa-analyzer` (or, `pip install wa-analyzer`)

This preprocesser uses the datetime module to convert strings with a date and / or time into datetime objects. The preprocessor needs to know the formatting of the timestamps in your `_chat.txt` file. Therefore, you might need to update the `datetime_format` variable in the `config.toml` file accordingly. You can find the formatting in the [documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) of the datetime module.

Now you can run the following command (on UNIX systems like Linux or OS X) to activate the virtual environment you have created in the previous step.

```bash
source .venv/bin/activate
```

You can check which python is being used by running:
```bash
which python
```

This should now return a path that includes the `.venv` folder in your project.
After this, you can run the preprocessor with the following command:

```bash
analyzer --device ios
```
Change `ios` to `android` if you have an android device.
This will run the `src/wa_analyzer.py:main` method, which will process the chat and save the results in the `data/processed` folder.

You should see some logs, like this:
```
2024-02-11 16:07:19.191 | INFO     | __main__:main:71 - Using iOS regexes
2024-02-11 16:07:19.201 | INFO     | __main__:process:61 - Found 1779 records
2024-02-11 16:07:19.201 | INFO     | __main__:process:62 - Appended 152 records
2024-02-11 16:07:19.202 | INFO     | __main__:save:30 - Writing to data/processed/whatsapp-20240211-160719.csv
2024-02-11 16:07:19.206 | SUCCESS  | __main__:save:32 - Done!
```

Inside the `log` folder you will find a logfile, which has some additional information that might be useful for debugging.

After this, put the name of the .csv file that is save to `inputpath` in the `config.toml` file.
You can then run the `notebooks/lesson1/01.3-your-own-chat.ipynb` notebook. This will save a cleaned `.parq` file. Put the name of that file after the `current` key in the `config.toml` file.

This `config.toml` file should make it easier to run the code with multiple input files; you can simply change the `current` value and run all notebooks for the file specified there.

# Run the graph dashboard
Once you have a processed file, explore the contact network interactively with:

```bash
uv run streamlit run streamlit_app.py
```

## codestyle
During the course, you will continue to improve your coding skills.
Use the [codestyle](https://github.com/raoulg/codestyle) repo as a reference!

## running the checks locally

Every notebook is expected to run top-to-bottom on the showcase data alone, with no
`config.toml` present — that's what CI checks on every push. To run the same checks
yourself:

```bash
uv run ruff check .                        # lint
uv run python tools/check_pipeline_drift.py  # lesson 1's ParseIRCLines vs scripts/pipelines.py
uv run notebooktester -f -t 900 notebooks/   # every notebook, top to bottom
```

`.lefthook.yml` already wires most of this into pre-commit hooks; `lefthook install`
(global install, not a project dependency) activates them, so a broken notebook is
caught before it reaches GitHub rather than after.



