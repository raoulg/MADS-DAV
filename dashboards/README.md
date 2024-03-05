# Dashboard Runner 🚀

Welcome to the Dashboard Runner project! This tool allows you to easily run different Streamlit dashboard scripts with a simple command. Perfect for quickly switching between various data visualizations or dashboards.

## Installation 🛠️

Before you start, ensure you have Python and `pdm` installed on your system. This project uses Streamlit, so you'll need to ensure you're working with the correct Python package versions.

1. **Pull the latest version from Git:** 🔄

   Ensure you have the latest version of the project to get the most recent `pyproject.toml` file, specifying version `0.2`, which includes Streamlit among other dependencies.

   ```bash
   git pull origin main
   ```

2. **Handling Merge Conflicts:** ❗

If you've made accidental changes to the main branch and encounter merge conflicts when pulling, follow these steps:

- Undo any changes to the main branch. Dont worry, we will save your changes in a new branch.
First, you need to find out what the commit hash if from the latest commit. You can do this by running:
  ```bash
  git log
  ```
  Then, you can reset the changes to the latest commit by running:
  ```bash
  git reset --hard <commit-hash>
  ```
  Replace `<commit-hash>` with the commit hash from the latest commit, this should look something like `2a5cb620`.

- Create and switch to a new branch:

  ```bash
  git checkout -b <your-new-branch-name>
  ```

- Add your changes:

  ```bash
  git add .
  ```

- Commit your changes:

  ```bash
  git commit -m "Describe your changes"
  ```

- Switch back to the main branch and pull the latest changes again:

  ```bash
  git checkout main
  git pull origin main
  ```

3. **Install Dependencies with PDM:** ⚙️

Once you have the latest version of the codebase and have resolved any conflicts, use `pdm` to install the required dependencies as specified in `pyproject.toml`.
```bash
pdm install
```
This ensures you have all the necessary dependencies installed in your environment.
Your pyproject.toml file should say "version = "0.2.0" in one of the first lines. If it doesn't, you may need to update your local copy of the repository.

# Usage 🚴

To run a specific dashboard script, use the `run_dashboard.sh` script. Here's how:

1. **Make the script executable (if not already done):** ✅

```bash
chmod +x run_dashboard.sh
```

2. Run your desired dashboard:
```bash
./run_dashboard.sh # Runs the default dashboard.py
./run_dashboard.sh 2 # Runs dashboard2.py
```
For additional help and options:
```bash
./run_dashboard.sh --help
```

## Contributing 🤝

Contributions to improve Dashboard Runner are always welcome! Whether it's adding new features, improving documentation, or reporting issues, feel free to make a pull request or open an issue.