This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation", previously known as "Data Mining & Exploration".
All instructions assume a UNIX machine. You should have received an invite link for a VM; if not, contact your teacher.
On the VM, everything is installed (like rye).

# Setup the virtual environment
1. First, make sure you have python >= 3.11. You can check the version with `python --version`.
2. Make sure `rye` is there
    - check if it is installed by executing `rye --help`
    - if not, run `curl -sSf https://rye.astral.sh/get | bash` (not necessary on the VM)
    - watch the intro video for rye at https://rye.astral.sh/guide/
3. Install the dependecies by navigating to the MADS-DAV folder where the `pyproject.toml` is located and run `rye sync`.

# Run the preprocessor

Download a chat from Whatsapp and put it in the `data/raw` folder. Rename the file to `chat.txt' and run the following command:

```bash
source .venv/bin/activate
```

This will activate your virtual environment.
You can check which python is being used by running:
```bash
which python
```

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

