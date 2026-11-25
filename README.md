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

## Raw data for the first run

Before running S0, copy the **500-account CSV supplied with the assignment** into the repository's `data/raw/` folder. The default first run expects this exact path relative to the repository root.

```text
data/raw/medium.csv
```

For the small sample, copy the supplied file to `data/raw/small.csv` and set `DATASET = "small"` in both S0 and S1. The full study uses the medium sample.

## Run the notebooks

Use the two pinned checkpoint download commands in [S2](notebooks/S2__selecting_models.ipynb) before running the foundation-model experiments. Run [S0](notebooks/S0__preprocessing.ipynb) through [S4](notebooks/S4__evaluation.ipynb) in order using the `.venv` kernel.

S0 writes the prepared inputs to `data/prepared/`. S3 writes the January forecasts and evaluation tables to `data/results/medium/2022-12-31/`. S4 reads those tables. The repository includes the folder structure. Git ignores the datasets and generated files inside it.

Keep account-level notebook outputs local when sharing the study.
