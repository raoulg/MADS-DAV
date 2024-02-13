This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation", previously known as "Data Mining & Exploration".

# Setup the virtual environment
1. First, make sure you have python >= 3.10. You can check the version with `python --version`.
2. Make sure you installed `pdm` , eg with `pip install pdm`.
3. Install the dependecies by navigating the the root where the `pyproject.toml` is located and run `pdm install`.

# Run the preprocessor

Download a chat from Whatsapp and put it in the `data/raw` folder. Rename the file to `chat.txt' and run the following command just once:

```bash
eval $(pdm venv activate)
```

This will activate your virtual environment. After this, you can run the preprocessor with the following command:

```bash
python src/preprocess.py --device ios
```
Change `ios` to `android` if you have an android device.


You should see some logs, like this:
```
2024-02-11 16:07:19.191 | INFO     | __main__:main:71 - Using iOS regexes
2024-02-11 16:07:19.201 | INFO     | __main__:process:61 - Found 1779 records
2024-02-11 16:07:19.201 | INFO     | __main__:process:62 - Appended 152 records
2024-02-11 16:07:19.202 | INFO     | __main__:save:30 - Writing to data/processed/whatsapp-20240211-160719.csv
2024-02-11 16:07:19.206 | SUCCESS  | __main__:save:32 - Done!
```

