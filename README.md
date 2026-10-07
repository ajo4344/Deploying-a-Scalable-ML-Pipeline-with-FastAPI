Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.






# Census Income Prediction API

This project trains a Random Forest model to predict whether annual
income is <=50K or >50K and serves predictions through FastAPI.

## Setup

Use Python 3.10.13. Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install flake8
```

## Train the Model

```bash
python train_model.py
```

The script loads data/census.csv, cleans surrounding spaces, and
creates an 80/20 train-test split. It saves the model and encoders
in model/ and categorical slice metrics in slice_output.txt.

Test performance is precision 0.7353, recall 0.6378, and F1 0.6831.
See model_card.md for details and limitations.

## Run Checks

```bash
python -m pytest -v
python -m flake8 .
```

Five tests cover training, inference, metrics, saving and loading,
unseen categories, and slice evaluation. GitHub Actions runs pytest
and flake8 on pushes and pull requests.

## Run the API

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

GET / returns a greeting. POST /data/ accepts census features and
returns a salary prediction.

Keep the server running and execute this in another terminal:

```bash
python local_api.py
```

## Screenshots

The screenshots/ folder contains:
- unit_test.png
- local_api.png
- continuous_integration.png
