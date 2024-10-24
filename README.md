# Setup

The requirements for scorepi are covered by the conda environment for this module.

Clone this repo with `git clone --recurse-submodules [PATH]`

Install conda environment from .yml file
`conda env create -f conda_requirements.yml`

Activate the conda environment
`conda activate epistorm-evaluations`

Then from the `scorepi` directory run
`pip install -e .`

If adding new package requirements or updating packages, it is best practice to place these in `conda_requirements.yml`, delete your existing environment, create a new environment from the YAML file, and then install scorepi with pip.

# Updating

When pulling new changes from the repo, add the `--recurse-submodules` flag to your `git fetch` or `git pull` calls.
