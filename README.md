This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation", previously known as "Data Mining & Exploration". All instructions assume a UNIX machine (Linux or Os X). You should have received an invite link for a linux VM; if not, contact your teacher. On the VM, everything is installed (like uv).

The manual for setting up the VM can be found in the `references` folder, in addition to a `git_crash_course` to help you work with git. Make sure to read both documents before you start asking questions about either.

# Setup the virtual environment
1. Make sure you have `uv` installed. On Unix systems, you can use `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. if the `MADS-DAV` folder isnt already cloned with git, add it with the command `git clone https://github.com/raoulg/MADS-DAV.git`
3. Navigate to the MADS-DAV folder where the `pyproject.toml` is located with `cd MADS-DAV` and run `uv sync --all-extras`. This should create a virtual environment (a .venv folder) and install all dependencies there.
4. Read through the [uv docs](https://docs.astral.sh/uv/guides/projects/) "working on projects" intro into `uv`

# Run the preprocessor
Download a chat from Whatsapp and put it in the `data/raw` folder. Rename the file to `chat.txt' and run the following command to activate the virtual environment you have created in the previous step.

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

## codestyle
During the course, you will continue to improve your coding skills. 
Use the [codestyle](https://github.com/raoulg/codestyle) repo as a reference!



