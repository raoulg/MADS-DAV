# Dashboard Runner 🚀
built with [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


Welcome to the Dashboard Runner project! This tool allows you to easily run different Streamlit dashboard scripts with a simple command. Perfect for quickly switching between various data visualizations or dashboards.

## Installation 🛠️
All dependencies are described by the `pyproject.toml` file. You can install everyting with
```bash
uv sync
```
or, if you insist on using the 10x slower pip, `pip install .`

# Usage 🚴

To run a specific dashboard script, use the `run_dashboard.sh` script. Here's how:

## Make the script executable

First, the script needs to be allowed to be executed. You can check this with the `ls -l` command:

```bash
❯ ls -l
total 24
-rw-rw-r-- 1 azureuser azureuser 1704 Sep  3 10:04 dashboard_1.py
-rw-rw-r-- 1 azureuser azureuser 2525 Sep  3 10:04 dashboard_2.py
-rw-rw-r-- 1 azureuser azureuser 2156 Sep  3 10:04 dashboard_3.py
-rw-rw-r-- 1 azureuser azureuser 2571 Sep  3 10:04 dashboard_4.py
-rw-rw-r-- 1 azureuser azureuser 1261 Oct 22 09:22 README.md
-rwxrwxr-x 1 azureuser azureuser 1092 Sep  3 10:04 run_dash.sh
```

Note that, for the `.sh` file, there are additonal `x`s which stand for `executable`.
If you dont see the x-es, you can run this command:

```bash
chmod +x run_dash.sh
```

## Run your desired dashboard:
You can now run the script. It will execute to python scripts.
```bash
./run_dash.sh # Runs the default dashboard.py
./run_dash.sh 2 # Runs dashboard2.py
```

Alternatively, you can do this manually by:
```bash
source .venv/bin/activate  # activating the environment
cd dashboards # cd-ing into the correct folder
streamlit run dashboard_1.py # executing the script
```

For additional help and options:
```bash
./run_dash.sh --help
```

## Open the dashboard
If you run this on a VM, VScode should automatically forward your port on `:8501`. You can see this under the tab `Ports`, next to the tab for `Terminal`.

Locally, you can open `http://localhost:8501/` and you should see your dashboard!

## Contributing 🤝
Contributions to improve Dashboard Runner are always welcome! Whether it's adding new features, improving documentation, or reporting issues, feel free to make a pull request or open an issue.
