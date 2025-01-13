import re
import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

#train.py script - Takes in the csv and writes our the final XGBoost model. This code was all originally in my notebook.ipynb

# Get the directory of the current script for proper input/output operations
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV
file_path = os.path.join(script_dir, 'data', 'rapid_only_games_metadata_profile_2024_01.csv')

# Load the data - this is a filtered down version that only includes rapid games from the full dataset in https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may/data dataset.
df = pd.read_csv(file_path)

df_clean = df.copy()

#Define critical columns that we will absolutely need
critical_columns = ['WhiteElo', 'BlackElo', 'White_playTime_total', 'Black_playTime_total', 'White_count_all', 'Black_count_all', 'Moves', 'TotalMoves', 'ECO', 'Opening']

#Drop empty rows for our critical columns
df_clean = df_clean.dropna(subset=critical_columns)

#Drop games that include a TOS violation
df_clean = df_clean[df_clean['White_tosViolation'].isnull() & df_clean['Black_tosViolation'].isnull()]

#Delete the tosViolation columns since we won't be needing them anymore
df_clean = df_clean.drop(columns=['White_tosViolation', 'Black_tosViolation'])

#Delete the date column since the day a particular game was played is not going to be used for this project
#We will also drop the profile_flag columns as that information can be arbitrarily set by the user and is not useful for prediction
df_clean = df_clean.drop(columns=['Date', 'White_profile_flag', 'Black_profile_flag'])


#Parse creation timestamps, and only take date of account creation
df_clean['White_createdAt'] = pd.to_datetime(df_clean['White_createdAt'], unit='ms')
df_clean['Black_createdAt'] = pd.to_datetime(df_clean['Black_createdAt'], unit='ms')
df_clean['White_createdAt'] = df_clean['White_createdAt'].dt.date
df_clean['Black_createdAt'] = df_clean['Black_createdAt'].dt.date

#Convert playtime to hours
df_clean['White_playTime_total'] = (df_clean['White_playTime_total'] / 3600).round(0).astype(int)
df_clean['Black_playTime_total'] = (df_clean['Black_playTime_total'] / 3600).round(0).astype(int)

# Rename columns for better clarity
df_clean = df_clean.rename(columns={
    'White_playTime_total': 'White_playTime_hours',
    'Black_playTime_total': 'Black_playTime_hours',
    'White_count_all': 'White_total_matches',
    'Black_count_all': 'Black_total_matches'
})


#Convert total game counts to int
df_clean['White_total_matches'] = (df_clean['White_total_matches']).astype(int)
df_clean['Black_total_matches'] = (df_clean['Black_total_matches']).astype(int)

# Remove games where either player has the BOT title
df_clean = df_clean[
    (df_clean['White_title'] != 'BOT') & 
    (df_clean['Black_title'] != 'BOT')
]

# Convert titles to ordinal values and add as columns to the cleaned dataframe
title_value = {
    'GM': 8,
    'IM': 7,
    'WGM': 6,
    'FM': 5,
    'NM': 4,
    'WIM': 4,
    'WFM': 3,
    'CM': 2,
    'WCM': 1,
    np.nan: 0,
    None: 0
}
white_title_idx = df_clean.columns.get_loc('White_title')
black_title_idx = df_clean.columns.get_loc('Black_title')
df_clean.insert(white_title_idx + 1, 'White_title_value', df_clean['White_title'].map(title_value))
df_clean.insert(black_title_idx + 1, 'Black_title_value', df_clean['Black_title'].map(title_value))


def parse_moves(dataframe):
    """
    Parse chess moves from a dataframe column and extract statistics by color.
    
    Parameters:
    dataframe (pandas.DataFrame): Input dataframe
    
    Returns:
    DataFrame with four new columns:
        - White_avgEval: Average evaluation after white moves
        - Black_avgEval: Average evaluation after black moves
        - White_avgMoveTime: Average time spent per move by white
        - Black_avgMoveTime: Average time spent per move by black
    """

    #Method to parse each move string per row
    def parse_game_moves(moves_str):

        moves = re.findall(r'(?:\d+\.{1,3}\s*)?([A-Za-z0-9#+=\-O]+\??!?!?\??)\s*{\s*(\[%eval[^]]+\])?\s*(\[%clk[^]]+\])?\s*}', moves_str)
        white_moves = moves[::2]
        black_moves = moves[1::2]
        
        # Helper function to extract evaluation value
        def extract_eval(eval_str):
            if not eval_str:
                return 0.0

            # Regular evaluation
            match = re.search(r'%eval\s*([-+]?\d*\.?\d*)', eval_str)
            # Only convert if we have a number
            if match and match.group(1):
                try:
                    return float(match.group(1))
                except ValueError:
                    return 0.0  # Return 0.0 for any parsing errors
            return 0.0  # Return 0.0 if no match found
        
        # Helper function to extract remaining clock time in seconds
        def extract_time(clock_str):
            if not clock_str:
                return None
            # Extract time from [%clk H:MM:SS] format
            match = re.search(r'%clk\s*(\d+):(\d+):(\d+)', clock_str)
            if match:
                hours, minutes, seconds = map(int, match.groups())
                return hours * 3600 + minutes * 60 + seconds
            return None
        
        def calculate_time_per_turn(times):
            if not times or len(times) < 2:
                return times

            # Calculate the time for the first move (assuming starting time was the previous player's clock)
            first_move_time = 0

            # Calculate times for subsequent moves
            subsequent_times = [times[i] - times[i+1] for i in range(len(times)-1) if times[i] is not None and times[i+1] is not None]

            return [first_move_time] + subsequent_times
        
        #Function to tally annotations https://en.wikipedia.org/wiki/Chess_annotation_symbols
        def tally_annotations(moves):
            brilliant_moves = 0
            good_moves = 0
            blunders = 0
            mistakes = 0

            def evaluate_annotation(annotation):
                if '!!' in annotation:
                    return 'brilliant'
                elif '!' in annotation and '?' not in annotation:
                    return 'good'
                elif '??' in annotation:
                    return 'blunder'
                elif '?' in annotation and '!' not in annotation:
                    return 'mistake'
                
            for move in moves:
                move_data = move[0]
                annotation = move_data[-2:]
                evaluation = evaluate_annotation(annotation)

                if evaluation == 'brilliant':
                    brilliant_moves += 1
                elif evaluation == 'good':
                    good_moves += 1
                elif evaluation == 'blunder':
                    blunders += 1
                elif evaluation == 'mistake':
                    mistakes += 1

            return brilliant_moves, good_moves, blunders, mistakes

        # Calculate statistics
        white_evals = [extract_eval(move[1]) for move in white_moves]
        black_evals = [-extract_eval(move[1]) for move in black_moves]
        white_times = [extract_time(move[2]) for move in white_moves]
        black_times = [extract_time(move[2]) for move in black_moves]

        #Tally annotations
        white_tally = tally_annotations(white_moves)
        white_brilliant_moves = white_tally[0]
        white_good_moves = white_tally[1]
        white_blunders = white_tally[2]
        white_mistakes = white_tally[3]

        black_tally = tally_annotations(black_moves)
        black_brilliant_moves = black_tally[0]
        black_good_moves = black_tally[1]
        black_blunders = black_tally[2]
        black_mistakes = black_tally[3]

        # Calculate time taken per turn in seconds
        white_seconds_per_turn = calculate_time_per_turn(white_times)
        black_seconds_per_turn = calculate_time_per_turn(black_times)
        
        # Calculate averages, filtering out None values
        white_avg_eval = sum(filter(None, white_evals)) / len(list(filter(None, white_evals))) if any(white_evals) else None
        black_avg_eval = sum(filter(None, black_evals)) / len(list(filter(None, black_evals))) if any(black_evals) else None
        white_avg_time = sum(filter(None, white_seconds_per_turn)) / len(list(filter(None, white_seconds_per_turn))) if any(white_seconds_per_turn) else None
        black_avg_time = sum(filter(None, black_seconds_per_turn)) / len(list(filter(None, black_seconds_per_turn))) if any(black_seconds_per_turn) else None
        
        return {
            'White_avgEval': white_avg_eval,
            'Black_avgEval': black_avg_eval,
            'White_avgMoveTime': white_avg_time,
            'Black_avgMoveTime': black_avg_time,
            'White_brilliantMoves' : white_brilliant_moves,
            'Black_brilliantMoves' : black_brilliant_moves,
            'White_goodMoves' : white_good_moves,
            'Black_goodMoves' : black_good_moves,
            'White_blunders' : white_blunders,
            'Black_blunders' : black_blunders,
            'White_mistakes' : white_mistakes,
            'Black_mistakes' : black_mistakes
        }
    
    # Apply the parsing function to each row
    results = []
    for _, row in dataframe.iterrows():
        try:
            results.append(parse_game_moves(row['Moves']))
        except Exception as e:
            print(f"Error processing row: {e}")
            results.append({
                'White_avgEval': None,
                'Black_avgEval': None,
                'White_avgMoveTime': None,
                'Black_avgMoveTime': None,
                'White_brilliantMoves' : None,
                'Black_brilliantMoves' : None,
                'White_goodMoves' : None,
                'Black_goodMoves' : None,
                'White_blunders' : None,
                'Black_blunders' : None,
                'White_mistakes' : None,
                'Black_mistakes' : None
            })
    
    # Convert results to DataFrame and join with original
    results_df = pd.DataFrame(results)
    return pd.concat([dataframe, results_df], axis=1, join='inner')

df_enriched = df_clean.copy()
df_enriched = parse_moves(df_enriched)


#Handle the shuffling and splitting of dataset

n = len(df_enriched)
n_val = int(n * 0.2)
n_test = int (n * 0.2)
n_train = n - n_val - n_test
n_val, n_test, n_train

#Shuffle the dataset:
np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

# Split the dataframe
df_train = df_enriched.iloc[idx[:n_train]]
df_val = df_enriched.iloc[idx[n_train:n_train+n_val]]
df_test = df_enriched.iloc[idx[n_train+n_val:]]

# Select numerical features that I have determined be good predictors of ELO
# (excluding the target variables and any non-numeric columns)
feature_columns = [
    'White_playTime_hours', 'White_total_matches', 'White_title_value',
    'Black_playTime_hours', 'Black_total_matches', 'Black_title_value',
    'TotalMoves', 'White_avgEval', 'Black_avgEval',
    'White_avgMoveTime', 'Black_avgMoveTime',
    'White_blunders', 'Black_blunders',
    'White_mistakes', 'Black_mistakes'
]

# Prepare training features
X_train = df_train[feature_columns].values
X_val = df_val[feature_columns].values
X_test = df_test[feature_columns].values

# Extract target variables (both White and Black ELO)
y_white = df_enriched['WhiteElo'].values
y_black = df_enriched['BlackElo'].values

# Split target variables
y_white_train = y_white[idx[:n_train]]
y_white_val = y_white[idx[n_train:n_train+n_val]]
y_white_test = y_white[idx[n_train+n_val:]]

y_black_train = y_black[idx[:n_train]]
y_black_val = y_black[idx[n_train:n_train+n_val]]
y_black_test = y_black[idx[n_train+n_val:]]

# Print shapes to verify the splits
print("Training set shape:", df_train.shape)
print("Validation set shape:", df_val.shape)
print("Test set shape:", df_test.shape)

print("\nWhite ELO splits shapes:")
print("Train:", y_white_train.shape)
print("Validation:", y_white_val.shape)
print("Test:", y_white_test.shape)

print("\nBlack ELO splits shapes:")
print("Train:", y_black_train.shape)
print("Validation:", y_black_val.shape)
print("Test:", y_black_test.shape)


# Create final DMatrix objects with all features
final_dtrain_white = xgb.DMatrix(X_train, label=y_white_train, feature_names=feature_columns)
final_dtrain_black = xgb.DMatrix(X_train, label=y_black_train, feature_names=feature_columns)

# Use the best parameters identified for both models
best_params_white = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'rmse'
}

best_params_black = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'rmse'
}

# Train final models
final_model_white = xgb.train(best_params_white, final_dtrain_white, num_boost_round=100)
final_model_black = xgb.train(best_params_black, final_dtrain_black, num_boost_round=100)


models_path = os.path.join(script_dir, 'models')
# Save models to files
with open(os.path.join(models_path,'xgb_model_white.bin'), 'wb') as f:
    pickle.dump(final_model_white, f)
with open(os.path.join(models_path,'xgb_model_black.bin'), 'wb') as f:
    pickle.dump(final_model_black, f)

# Save feature columns for later reference
with open(os.path.join(models_path,'feature_columns.bin'), 'wb') as f:
    pickle.dump(feature_columns, f)

print("Final Models and feature columns have been saved to models directory as pickle binaries!")