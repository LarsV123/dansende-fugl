# Dansende fugl

This repository holds the code used by our group ("Dansende fugl") for the course "TMA4851 - Experts in Teamwork - Data deluge - what can we learn from an abundance of data?" in 2022.
The code was used to generate the results used in our reports for the course.
Our project used a data set provided by Matej Gjurković and Jan Šnajder (https://takelab.fer.hr/data/mbti/).

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
- Start Postgres with `docker-compose --compatibility up`
- Load the dataset(s) with the CLI tool
- Create a `.env` file based on `.env-template`

## Command line interface (CLI)

The command line interface in `cli.py` can be used to run some commands, such
as inserting the dataset(s). Example commands:

```python
# Load all datasets
python cli.py init
```
