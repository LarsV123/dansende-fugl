# Dansende fugl

TMA4851 - Repo for EiT group "Dansende Fugl"

## Contributors

- Lars-Olav Vågene
- Sander Francis
- Ingvild Devold
- Torje Nysæther
- Mihajlo Krivokapic

## Setup

Prerequisites:

- Python
- Docker

Installation steps:

```python
# Set up virtual environment (optional)
python -m venv venv

# Install dependencies
pip install -r requirements.txt
```

Recommended steps:

- Set up your IDE to automatically reformat files with Black (or run `black .` manually before committing).
- Download dataset and add it to the `/data` folder.
- Start Postgres with `docker-compose up`
- Load the dataset(s) with the CLI tool
- Create a `.env` file based on `.env-template`

## Command line interface (CLI)

The command line interface in `cli.py` can be used to run some commands, such
as inserting the dataset(s). Example commands:

```python
# Load all datasets
python cli.py init
```
