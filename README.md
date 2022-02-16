# Dansende fugl

TMA4851 - Repo for EiT group "Dansende Fugl"

## Contributors

- Lars-Olav Vågene
- Sander Francis
- Ingvild Devold
- Torje Nysæther

## Setup

Prerequisites:

- Python

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
- Load the dataset(s) with the CLI tool

## Command line interface (CLI)

The command line interface in `cli.py` can be used to run some commands, such
as inserting the dataset(s) into an SQLite database. Example commands:

```python
# Load all datasets
python cli.py init

# Create indexes
python cli.py index

# Check that the data has been loaded
python cli.py sample --table typed_posts

# Sample commands to demonstrate
python cli.py hello --help
python cli.py hello Bob --count 4
```
