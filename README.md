# Epistorm Evaluations

This repo contains automatically updated evaluations for all models participating in the [FluSight forecast hub](https://github.com/cdcepi/FluSight-forecast-hub) from the 2021-2022 season through the present. Evaluations will always use the most recent versions/revisions of surveillance and forecast numbers.

Up-to-date evaluations are available in `/evaluations`. **This directory is load-bearing** - it is used for visualizations in [epistorm-dashboard](https://github.com/mobs-lab/epistorm-dashboard) hosted at https://fluforecast.epistorm.org/.

**DO NOT MAKE CHANGES TO THE `/evaluations` DIRECTORY WHICH YOU DO NOT WANT REFLECTED DOWNSTREAM**

Predictions for the 2021-2022 and 2022-2023 seasons come from a separate [archive](https://github.com/cdcepi/Flusight-forecast-data). The archive is scored separately by the `archive_evaluations` workflow, which saves scores to `/evaluations/archive-2021-2023` and inserts them into the main evaluations files in `/evaluations`.

## Running Scratch Evaluations via GitHub Actions

The `scratch_evalations` workflow enables you to manually initiate evaluations with the following inputs:

Models: either `all` or specify any number of models by name, space-separated in a single string without quotes (defaults to `MOBS-GLEAM_FLUH`).

Dates: either `all` or specify any number of dates in YYYY-MM-DD format, space-separated in a single string without quotes (defaults to `all`).

This workflow outputs evaluations for the specified models and dates to the `/scratch` directory, overwriting existing files. Run the workflow via the GUI in the GitHub Actions tab.

This workflow does not score archived seasons (2021-2023). It uses the most up-to-date data from [FluSight forecast hub](https://github.com/cdcepi/FluSight-forecast-hub) at time of initiation.

## Up-To-Date Evaluations

The `update_evaluations` workflow runs on a schedule to provide updated evaluations in the `/evaluations` directory. You can manually initiate an out-of-schedule update via the GitHub Actions tab.

The update schedule is:

- Every Thursday, starting from 6:00 AM UTC until 9:00 PM UTC, every 3 hours
- Every Wednesday, Friday, and Saturday at 9:00 PM UTC

## Running Locally

Install conda environment from .yml file
`conda env create -f conda_requirements.yml`
and activate the environment
`conda activate epistorm-evaluations`

Open `epistorm_evaluations.ipynb` with your preferred editor, e.g. `jupyter lab`.

Alternatively, run `python epistorm_evaluations.py --mode scratch --models ... --dates ...`, replacing ellipses with desired inputs. 

## File Directory

- `epistorm_evaluations.ipynb` is a working evaluations notebook.
- `epistorm_evaluations.py` is the automated evaluations script.
- `.github/workflows/update_evaluations.yml` runs `epistorm_evaluations.py` in update mode on a schedule or on manual initiation, using `data_retrieval.sh` to track updates, and uploads the new evaluations to `/evaluations`. This workflow is responsible for maintaining up-to-date evaluations.
- `.github/workflows/scratch_evaluations.yml` runs `epistorm_evaluations.py` in scratch mode for the specified models and dates. Results are uploaded to `/scratch` and overwrite existing files in this directory.
- `.github/workflows/archive-evaluations.yml` runs `epistorm_evaluations.py` on 2021-2023 archive data. Results are uploaded to `/evaluations/archive-2021-2023` as well as inserted into the main evaluations files in `/evaluations`.
- `/evaluations` contains up-to-date evaluations of all models.
- `/evaluations/archive-2021-2023` contains evaluations for archive seasons.
- `/scratch` contains scratch evaluations output.
- `/Flusight-forecast-hub` is the submodule repo containing all data.
- `/data` contains copied data for use in automated evaluations with update tracking.
- `/deprecated` contains old versions of files.
- `data_retrieval.sh` copies and tracks updated data for use in update workflow.
- `updated_forecasts.csv` contains paths to forecast files which have been updated since the last evaluations run, as recorded by `data_retrieval.sh`.
- `conda_requirements.yml` for running locally with a conda environment.
- `pip_requirements.txt` for running on GitHub Actions (or locally) with pip.
