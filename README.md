# Bank balance forecasting

This study forecasts the next 30 daily balances for each of 500 accounts. The notebooks follow the study from preprocessing through evaluation.

## Set up Python locally

Install Python 3.12 and open a terminal in the repository folder. Create and activate the project-local `.venv` using the commands for your shell.

**macOS or Linux**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

With the environment active, install all project dependencies, including Chronos, and start JupyterLab.

```sh
python -m pip install -e ".[chronos]"
python -m jupyter lab
```

## Run the notebooks

Put the supplied `medium.csv` in `data/raw/`. S0 and S1 also accept `small.csv`. Use the two pinned checkpoint download commands in [S2](notebooks/S2__selecting_models.ipynb) before running the foundation-model experiments.

Run [S0](notebooks/S0__preprocessing.ipynb) through [S4](notebooks/S4__evaluation.ipynb) in order using the `.venv` kernel. S0 writes the prepared inputs to `data/prepared/`. S3 writes the January forecasts and evaluation tables to `data/results/medium/2022-12-31/`. S4 reads those tables. The data directory stays local and is ignored by Git.

Keep account-level notebook outputs local when sharing the study.
