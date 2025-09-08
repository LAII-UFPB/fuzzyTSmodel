# FuzzyTSmodel

Fuzzy model with automatic rules generation for time series forecasting.

This package is an interface to the [simpful library](https://github.com/aresio/simpful/tree/master) intended to be used in a time series forecasting context.
See `sin_exemple.py` to learn how to use it.

## Clone the repository

First, clone this repository from GitHub:

```bash
git clone https://github.com/LAII-UFPB/fuzzyTSmodel.git
cd fuzzyTSmodel
```

## Installation
Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
uv pip install -e .
```

If you don't have `uv` installed, run:
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or 

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

but if you prefer, you can use pip.
```bash
pip install -e .
pip install -r requirements.txt
```

## Class Diagram

![FuzzyTSmodel Class UML](images/UML%20class.png "FuzzyTSmodel class UML")