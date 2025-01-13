import streamlit as st
import requests
import json
import os
import numpy as np

# Sample generated game data
SAMPLE_GAMES = {
   "Grandmaster Rapid Game": {
       "features": {
           "White_playTime_hours": 200.0,
           "White_total_matches": 50000,
           "White_title_value": 8,
           "Black_playTime_hours": 180.0,
           "Black_total_matches": 45000,
           "Black_title_value": 7,
           "TotalMoves": 45,
           "White_avgEval": 0.5,
           "Black_avgEval": -0.5,
           "White_avgMoveTime": 10.0,
           "Black_avgMoveTime": 9.8,
           "White_blunders": 1,
           "Black_blunders": 1,
           "White_mistakes": 2,
           "Black_mistakes": 2
       }
   },
   "Expert Rapid Game": {
       "features": {
           "White_playTime_hours": 50.0,
           "White_total_matches": 5000,
           "White_title_value": 2,
           "Black_playTime_hours": 40.0,
           "Black_total_matches": 4000,
           "Black_title_value": 1,
           "TotalMoves": 35,
           "White_avgEval": -1.5,
           "Black_avgEval": 1.5,
           "White_avgMoveTime": 15.0,
           "Black_avgMoveTime": 14.5,
           "White_blunders": 5,
           "Black_blunders": 4,
           "White_mistakes": 6,
           "Black_mistakes": 5
       }
   },
   "National Master Rapid Game": {
       "features": {
           "White_playTime_hours": 80.0,
           "White_total_matches": 8000,
           "White_title_value": 4,
           "Black_playTime_hours": 70.0,
           "Black_total_matches": 7000,
           "Black_title_value": 2,
           "TotalMoves": 38,
           "White_avgEval": 0.3,
           "Black_avgEval": -0.3,
           "White_avgMoveTime": 11.5,
           "Black_avgMoveTime": 11.0,
           "White_blunders": 3,
           "Black_blunders": 4,
           "White_mistakes": 4,
           "Black_mistakes": 5
       }
   },
   "Online Casual Rapid Game": {
       "features": {
           "White_playTime_hours": 20.0,
           "White_total_matches": 1500,
           "White_title_value": 1,
           "Black_playTime_hours": 15.0,
           "Black_total_matches": 1200,
           "Black_title_value": 0,
           "TotalMoves": 32,
           "White_avgEval": -1.0,
           "Black_avgEval": 1.0,
           "White_avgMoveTime": 18.0,
           "Black_avgMoveTime": 17.5,
           "White_blunders": 6,
           "Black_blunders": 7,
           "White_mistakes": 8,
           "Black_mistakes": 9
       }
   },
   "Veteran Fide Master Rapid Game": {
       "features": {
           "White_playTime_hours": 300.0,
           "White_total_matches": 25000,
           "White_title_value": 5,
           "Black_playTime_hours": 280.0,
           "Black_total_matches": 22000,
           "Black_title_value": 4,
           "TotalMoves": 42,
           "White_avgEval": 0.1,
           "Black_avgEval": -0.1,
           "White_avgMoveTime": 13.0,
           "Black_avgMoveTime": 12.5,
           "White_blunders": 2,
           "Black_blunders": 2,
           "White_mistakes": 3,
           "Black_mistakes": 3
       }
   }
}

def validate_input(input_dict):
    """
    Validate and sanitize user input to prevent potential security issues
    """
    required_keys = [
        "White_playTime_hours", "White_total_matches", "White_title_value",
        "Black_playTime_hours", "Black_total_matches", "Black_title_value",
        "TotalMoves", "White_avgEval", "Black_avgEval", 
        "White_avgMoveTime", "Black_avgMoveTime", 
        "White_blunders", "Black_blunders", 
        "White_mistakes", "Black_mistakes"
    ]
    
    # Ensure all required keys are present
    if not all(key in input_dict for key in required_keys):
        raise ValueError("Missing required input features")
    
    # Validate and convert to float/int
    sanitized_input = {}
    try:
        for key in required_keys:
            # Convert to float for numeric keys
            if any(x in key for x in ['hours', 'Moves', 'avgEval', 'avgMoveTime']):
                val = float(input_dict[key])
                # Add reasonable bounds checking
                if key.endswith('hours') and (val < 0 or val > 1000):
                    raise ValueError(f"Invalid {key} value")
                if 'avgEval' in key and abs(val) > 10:
                    raise ValueError(f"Invalid {key} value")
                sanitized_input[key] = val
            # Convert to int for count-based keys
            else:
                val = int(input_dict[key])
                if val < 0:
                    raise ValueError(f"Invalid {key} value")
                sanitized_input[key] = val
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input: {str(e)}")
    
    return sanitized_input

def main():
    st.title("Chess ELO Rating Predictor")
    st.write("https://github.com/mar1-k/chess_elo_prediction")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:", 
        ["Sample Rapid Games", "Manual Input"]
    )
    
    if input_method == "Sample Rapid Games":
        # Select sample game
        selected_game = st.selectbox(
            "Choose a generated sample rapid game: ",
            list(SAMPLE_GAMES.keys())
        )
        game_data = SAMPLE_GAMES[selected_game]
    else:
        # Manual input form
        st.write("### Manual Input of Game Features")
        
        # Create input columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**White Player Features**")
            white_play_time = st.number_input("Play Time (hours)", min_value=0.0, value=10.0, step=1.0)
            white_total_matches = st.number_input("Total Matches", min_value=0, value=500)
            white_title_value = st.selectbox("White Title Value", [0, 1], index=0, key="white_title_value")
            white_avg_eval = st.number_input("White Average Evaluation", value=0.0, step=0.1, key="white_avg_eval")
            white_avg_move_time = st.number_input("White Average Move Time (seconds)", min_value=0.0, value=5.0, step=0.5, key="white_avg_move_time")
            white_blunders = st.number_input("White Blunders", min_value=0, value=2, key="white_blunders")
            white_mistakes = st.number_input("White Mistakes", min_value=0, value=3, key="white_mistakes")
        
        with col2:
            st.write("**Black Player Features**")
            black_play_time = st.number_input("Black Play Time (hours)", min_value=0.0, value=8.0, step=1.0, key="black_play_time")
            black_total_matches = st.number_input("Black Total Matches", min_value=0, value=450, key="black_total_matches")
            black_title_value = st.selectbox("Black Title Value", [0, 1], index=0, key="black_title_value")
            black_avg_eval = st.number_input("Average Evaluation", value=0.0, step=0.1)
            black_avg_move_time = st.number_input("Average Move Time (seconds)", min_value=0.0, value=4.5, step=0.5)
            black_blunders = st.number_input("Blunders", min_value=0, value=2)
            black_mistakes = st.number_input("Mistakes", min_value=0, value=3)
        
        # Total moves input
        total_moves = st.number_input("Total Moves in Game", min_value=1, value=35)
        
        # Prepare game data for manual input
        game_data = {
            "features": {
                "White_playTime_hours": white_play_time,
                "White_total_matches": white_total_matches,
                "White_title_value": white_title_value,
                "Black_playTime_hours": black_play_time,
                "Black_total_matches": black_total_matches,
                "Black_title_value": black_title_value,
                "TotalMoves": total_moves,
                "White_avgEval": white_avg_eval,
                "Black_avgEval": black_avg_eval,
                "White_avgMoveTime": white_avg_move_time,
                "Black_avgMoveTime": black_avg_move_time,
                "White_blunders": white_blunders,
                "Black_blunders": black_blunders,
                "White_mistakes": white_mistakes,
                "Black_mistakes": black_mistakes
            }
        }
    
    # Predict button
    if st.button("Predict ELO Ratings"):
        try:
            # Validate input
            validated_features = validate_input(game_data["features"])
            
            # Prepare request
            predict_api_url = os.environ.get('PREDICTION_SERVICE_URL', 'http://host.docker.internal:8000') #This is changed to a service URL in cloud deployment via this env variable, otherwise default to docker
            predict_api_url = predict_api_url + '/predict'
            response = requests.post(
                predict_api_url,
                json={"features": validated_features},
                timeout=5
            )

            if response.status_code == 200:
                predictions = response.json()
                st.success("Model Predictions:")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("White ELO", f"{predictions['white_elo']:.0f}")
                with col2:
                    st.metric("Black ELO", f"{predictions['black_elo']:.0f}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to prediction service: {str(e)}")
        except ValueError as e:
            st.error(f"Input validation error: {str(e)}")

    # Display game features
    st.write("### Game Features")
    st.json(game_data)

    # Add some explanation
    st.write("### How it works")
    st.write("""
    **Note: This model is trained specifically on rapid chess games**

    The ELO rating prediction is based on game statistics including:
    - Number of moves
    - Average evaluation
    - Mistakes and blunders
    - Play time and move times
    - Player experience (total matches)

    Predictions are made using XGBoost models trained on a large dataset of analyzed rapid chess games from Lichess (https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may/data).

    **Important Considerations:**
    - The model may have reduced accuracy for game types significantly different from rapid games
    - Ratings are estimates based on game performance metrics
    - Individual game performance can vary widely
    """)

if __name__ == "__main__":
    main()