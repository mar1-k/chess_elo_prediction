https://console.developers.google.com/apis/api/cloudresourcemanager.googleapis.com/overview?project=chess-elo-prediction
https://console.developers.google.com/apis/api/run.googleapis.com/overview?project=chess-elo-prediction
https://console.developers.google.com/apis/api/compute.googleapis.com/overview?project=chess-elo-prediction

gcloud config set project chess-elo-prediction
gcloud auth login
gcloud auth configure-docker


docker build -t elo-predictor .
docker tag elo-predictor:latest gcr.io/chess-elo-prediction/elo-prediction-api:latest
docker push gcr.io/chess-elo-prediction/elo-prediction-api:latest


docker build -t elo-frontend . 
docker tag elo-frontend:latest gcr.io/chess-elo-prediction/predict-chess-elo-streamlit-frontend:latest
docker push gcr.io/chess-elo-prediction/predict-chess-elo-streamlit-frontend:latest