# Epistorm Evaluations

This repo contains automatically updated evaluations for all models participating in the [FluSight forecast hub](https://github.com/cdcepi/FluSight-forecast-hub).

Up-to-date evaluations are available in `/evaluations`. **This directory is load-bearing** - it is used for visualizations in [epistorm-dashboard](https://github.com/mobs-lab/epistorm-dashboard) hosted at https://fluforecast.epistorm.org/.

**DO NOT MAKE CHANGES TO THE `/evaluations` DIRECTORY WHICH YOU DO NOT WANT REFLECTED DOWNSTREAM**

## Directory

- `epistorm_evaluations.ipynb` is a working evaluations notebook.
- `epistorm_evaluations.py` is the automated evaluations script.
- `.github/workflows/update_evaluations.yml` runs `epistorm_evaluations.py` in update mode on a schedule or on manual initiation and uploads the new evaluations to `/evaluations`. This workflow is responsible for maintaining up-to-date CI/CD evaluations of all models for all time.
- `.github/workflows/update_evaluations.yml` runs `epistorm_evaluations.py` in scratch mode for the specified models and dates. Results are upoaded to `/scratch` and overwrite existing files in this directory.
- `/evaluations` contains up-to-date evaluations of all models for all time.
- `/scratch` contains scratch evaluations output.
- `/Flusight-forecast-hub` is the submodule repo containing all data.
- `/data` contains copied data for use in automated evaluations with update tracking.
- `/deprecated` contains old versions of files.
- `data_retrieval.sh` copies and tracks updated data for use in CI/CD workflow.
- `updated_forecasts.csv` contains paths to forecast files which have been updated since the last evaluations run, as recorded by `data_retrieval.sh`.
- `conda_requirements.yml` for running locally with a conda environment.
- `pip_requirements.txt` for running on GitHub Actions (or locally) with pip.

## Running Scratch Evaluations via GitHub Actions

The `scratch_evalations` workflow enables you to manually initiate evaluations with the following inputs:

Models: either `all` or specify any number of models by name, space-separated in a single string without quotes (defaults to `MOBS-GLEAM_FLUH`).

Dates: either `all` or specify any number of dates in YYYY-MM-DD format, space-separated in a single string without quotes (defaults to `all`).

This workflow outputs evaluations for the specified models and dates to the `/scratch` directory, overwriting existing files. Run the workflow via the GUI in the GitHub Actions tab.

## Up-To-Date Evaluations for All Models for All Time

The `update_evaluations` workflow runs on a schedule to provide updated evaluations in the `/evaluations` directory. You can manually initiate an out-of-schedule update via the GitHub Actions tab.

## Running Notebook Locally

Install conda environment from .yml file
`conda env create -f conda_requirements.yml`
and activate the environment
`conda activate epistorm-evaluations`

Open `epistorm_evaluations.ipynb` with your preferred editor, e.g. `jupyter lab`.
