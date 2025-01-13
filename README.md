# Chess ELO Rating Predictor

Access the live demo at https://predict-chess-elo-streamlit-frontend-zsmu2mtvwa-uc.a.run.app

## Problem Description

This project implements a machine learning solution to predict the ELO ratings of both White and Black players in chess games based on their gameplay statistics and performance metrics. The model analyzes various aspects of a chess game, including player experience, game duration, move quality, and decision-making patterns to estimate players' skill levels (ELO ratings).

### Why This Is Useful
- **Player Skill Assessment**: Helps evaluate a player's true skill level based on their gameplay patterns rather than just win/loss records
- **Performance Analysis**: Provides insights into how different aspects of gameplay correlate with player ratings
- **Skill Development**: Can be used to track progress and identify areas for improvement in a player's game

## Dataset

The project uses a filtered subset of the [Lichess Games Dataset](https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may/data) from 2024, focusing specifically on rapid games. The filitered dataset includes approximately 30,000 games with detailed move analysis and player statistics.

### Key Features
- Player Statistics (White/Black)
  - Total playtime (hours)
  - Total matches played
  - Title value (GM=8, IM=7, etc.)
- Game Analysis
  - Total moves in game
  - Average position evaluation
  - Average move time
  - Number of blunders
  - Number of mistakes

## Model Performance

The project evaluated several machine learning models and configuration, with XGBoost showing the best performance:

| Model | White ELO RMSE | Black ELO RMSE |
|-------|---------------|----------------|
| Linear Regression | 323.11 | 326.59 |
| Decision Tree | 296.58 | 297.53 |
| Random Forest | 290.07 | 291.90 |
| XGBoost | 287.89 | 290.71 |

See notebook.ipynb for full analysis

## Model Details

The final implementation uses XGBoost with the following configurations:

### White ELO Model
- Learning rate: 0.1
- Max depth: 3
- Subsample: 0.9
- Colsample_bytree: 0.9

### Black ELO Model
- Learning rate: 0.3
- Max depth: 3
- Subsample: 0.9
- Colsample_bytree: 0.9


### Technology and Libraries Used
```
- python 3.11
- pandas
- matplotlib
- numpy
- scikit-learn
- xgboost
- fast api
- streamlit
- Docker
- Google Cloud Platform

```

### Environment Setup

1. Clone the repo and make and navigate into it
```
git clone https://github.com/mar1-k/chess_elo_prediction.git
cd chess_elo_prediction
```

2. Install Pipenv:
```
pip install pipenv
```

3. Install dependencies:
```
pipenv install
```

4. Activate the virtual environment:
```
pipenv shell
```

## Project Structure

```
chess_elo_prediction/
├── README.md
├── data/
│   └── rapid_only_games_metadata_profile_2024_01.csv
├── deployment/
│   ├── README.md
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
├── models/
│   ├── feature_columns.bin
│   ├── xgb_model_black.bin
│   └── xgb_model_white.bin
├── predict_api/ #Model files have been placed in this directory as well for ease of dockerizing
│   ├── dockerfile
│   ├── feature_columns.bin 
│   ├── predict.py
│   ├── requirements.txt
│   ├── xgb_model_black.bin
│   └── xgb_model_white.bin
├── streamlit_frontend/
│   ├── app.py
│   ├── dockerfile
│   └── requirements.txt
├── notebook.ipynb
├── Pipfile
├── Pipfile.lock
├── README.md
└── train.py
```

## Usage Instructions

### Running the notebook
1. Ensure the dataset `rapid_only_games_metadata_profile_2024_01.csv` is in the `data/` directory
2. Open and run notebook.ipynb - ensure that you have installed all dependencies from environment setup
```
jupyter notebook notebook.ipynb
```
3. Run the notebook

### Training the Model

1. Ensure the dataset `rapid_only_games_metadata_profile_2024_01.csv` is in the `data/` directory
2. Run the training script:
```
python train.py
```
This will generate the model binary picke files and place them in the `models/` directory.

### Running the Models locally
The application is containerized using Docker for consistent deployment across different environments.

1. Deploy the predict_api service:
```
#Navigate to the predict_api folder of this project
cd predict_api

# Build the Docker image - this will use the binary files found locally in the directory
docker build -t chess-elo-predictor .

# Run the container
docker run -p 8000:8000 chess-elo-predictor  
```

2. Deploy the Streamlit front end service (Optional)
```
#Navigate to the streamlit_frontend folder of this project
cd streamlit_frontend

# Build the Docker image
docker build -t chess-elo-frontend . 

# Run the container
docker run -p 8501:8501 chess-elo-frontend 
```

3. Navigate to the service hosted at localhost:8501 or use the API hosted at localhost:8000 directly


## Deployment

The application is deployed using Docker and Google Cloud Run. Access the live demo at https://predict-chess-elo-streamlit-frontend-zsmu2mtvwa-uc.a.run.app

### Cloud Deployment

The application is deployed on Google Cloud Run, providing scalable and serverless execution. This is faciliated through Terraform for ease of reproducibility and maintaining IaC. 

Deployment instructions:

1. Create a GCP cloud project

2. Create a Terraform Service Account and grant it appropriate permissions - Editor and Cloud Run Admin

3. Create a Service Account Key
- From the console, click on your newly created service account and navigate to the "KEYS" tab
- Click on "Add Key" to Create a key file for this service account
- Save the key file somewhere safe and accessible on the system that you will be using Terraform from

4. Enable Necessary APIs
Terraform will need the following GCP APIs enabled for this project, please enable them in your project
https://console.developers.google.com/apis/api/cloudresourcemanager.googleapis.com
https://console.developers.google.com/apis/api/run.googleapis.com
https://console.developers.google.com/apis/api/compute.googleapis.com
https://console.developers.google.com/apis/api/vpcaccess.googleapis.com

5. Setup Terraform Variables file 
- Navigate to the Terraform folder of this project and ensure that the Terraform variables file `variables.tf` has the correct project name and GCP key file path information

6. Push Docker containers to GCR
```
gcloud config set project <YOUR PROJECT NAME>
gcloud auth login
gcloud auth configure-docker

#In the /predict_api directory
docker build -t chess-elo-predictor .
docker tag chess-elo-predictor:latest gcr.io/chess-elo-prediction/elo-prediction-api:latest
docker push gcr.io/chess-elo-prediction/elo-prediction-api:latest

#In the /streamlit_frontend directory
docker build -t chess-elo-frontend . 
docker tag chess-elo-frontend:latest gcr.io/chess-elo-prediction/predict-chess-elo-streamlit-frontend:latest
docker push gcr.io/chess-elo-prediction/predict-chess-elo-streamlit-frontend:latest
```

7. Terraform init and apply
- While in the Terraform folder of this project run `terraform init` and then `terraform apply`
- Review the Terraform plan and type `yes` if everything looks good, you should see `Plan: 18 to add, 0 to change, 0 to destroy.

8. Navigate to the frontend url provided by Terraform

9. Enjoy your deployment! Don't forget to `terraform destroy` when done

## Acknowledgments

This has been a capstone project for the 2024-2025 cohort of DataTalks.Club Machine Learning Zoomcamp. I am once again deeply grateful to Alexey Grigorev and the team for making this quality course available completely free and painstaklingly going through the effort of making it all possible.

I am also thankful to
- Lichess.org for the original dataset
- Kaggle dataset curator Shkarupylo Maxim
