# Steniq: Topological Data Analysis -  January 2024

## Installation using ```poetry```

We use [```poetry```](https://python-poetry.org/docs/), a tool for dependency management and packaging in ```Python``` to create a virtual environment (```venv```) from which run the project. You can install ```poetry``` following the steps described [here](https://python-poetry.org/docs/#installation).

Clone the repository using:

```bash
git clone git@github.com:jmiramont
```

Then, simply let ```poetry``` do all the work. Create a virtual environment and install all the current dependencies using:

```bash
poetry install
```

*Remark for conda users:*

*If you have [`Anaconda`](https://www.anaconda.com/) or [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html) installed please disable the auto-activation of the base environment and your conda environment using:*

```bash
conda config --set auto_activate_base false
conda deactivate
```
