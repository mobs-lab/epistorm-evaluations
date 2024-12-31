# Epistorm Evaluations

This repo contains automatically updated evaluations for all models participating in the [FluSight forecast hub](https://github.com/cdcepi/FluSight-forecast-hub).

Up-to-date evaluations are available in `/evaluations`. **This directory is load-bearing** - it is used for visualizations in [epistorm-dashboard](https://github.com/mobs-lab/epistorm-dashboard) hosted at https://fluforecast.epistorm.org/.

**DO NOT MAKE CHANGES TO THE `/evaluations` DIRECTORY WHICH YOU DO NOT WANT REFLECTED DOWNSTREAM**

## Directory

- `epistorm_evaluations.ipynb` is a working evaluations notebook
- `epistorm_evaluations.py` is the automated evaluations script
- `.github/workflows/evaluate_predictions.yml` runs `epistorm_evaluations.py` on a schedule or on manual initiation and uploads the new evaluations to `/evaluations`.
- `/dat` contains data for use in evaluations
- `/deprecated` contains old versions of files
- `conda_requirements.yml` for running locally with a conda environment
- `pip_requirements.txt` for running on GitHub Actions (or locally) with pip

## Manually Running via GitHub Actions

The `evaluate-predictions` workflow enables you to manually initiate evaluations with the following inputs:

Mode: `most_recent` only evaluates predictions with a reference date equalling the most recent surveillance date, while `recalculate_all` overwrites existing data and evaluates all selected models for all dates.

Output Directory: Writing to the `scratch` directory always overwrites existing files. Use this directory for testing and exploration. The `evaluations` directory should only ever contain non-duplicate up-to-date evaluations for all models and all dates.

Models: specify any number of models by name, space-separated in a single string (defaults to all models).

## Running Locally

Install conda environment from .yml file
`conda env create -f conda_requirements.yml`

Open `epistorm_evaluations.ipynb` with your preferred editor, e.g. `jupyter lab`
