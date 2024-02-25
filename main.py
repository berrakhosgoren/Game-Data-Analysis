import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# SETUP

main_folder = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-3\Python'
data_folder = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-3\Meta-data'

# List all files in the folder
file_names = [file for file in os.listdir(data_folder) if file.endswith('.parquet')]

# remove data 11,12, 18, 19 and 5 from the source files and analyze them separately
# level of the game is not certain in these datasets
files_to_exclude = ['data_11.parquet', 'data_12.parquet','data_5.parquet','data_18.parquet','data_19.parquet']  

# construct file paths excluding the specified files
file_paths = [
    os.path.join(data_folder, file)
    for file in file_names
    if file not in files_to_exclude
]


## STEP 01: ANALYZE THE DATA

# event schema:
# "1": "user pressed green button and the character was a match",
# "2": "user pressed red button and the character was not a match",
# "3": "user pressed green button and the character was not a match",
# "4": "user pressed red button and the character was a match",
# "5": "user selected no answer",
# "6": "character presentation"

# 3 Levels, Each level: 25 charecter presentation

all_score = [] # empty list to store avg scores of all players
all_time  = [] # empty list to store avg time of all players

# loop over all data files
for file_path in file_paths:
    
    game_data = pd.read_parquet(file_path)
    
    # Extraction of 2 features is needed:
    # Score: Right answer
    # Time:  Time to answer 
    
    # separete the data into levels 
    level1_data = game_data.iloc[0:50] # in each level there are 50 rows (25 x 2)
    level2_data = game_data.iloc[50:100]
    level3_data = game_data.iloc[100:150]
    
    # store levels in a list
    levels = [level1_data, level2_data, level3_data]
    
    score_matrix = [] # create a empty score list for each file
    time_matrix  = [] # create a empty time list for each file
    
    # loop over levels
    for level_data in levels:
        
        
        # Calculate the total correct answer in the level
        #------------------------------------------------
        count1_idx = level_data[level_data['event_id'] == 1].index # find event 1 indicies
        count1 = len(count1_idx) # count
        count2_idx = level_data[level_data['event_id'] == 2].index # find event 2 indicies
        count2 = len(count2_idx) # count
         
        score_level = count1 + count2 # calculate the score
        score_matrix.append(score_level) # columns: level1, level2,level3
        
    
        # Calculate the total time taken to answer in each level
        #-------------------------------------------------------
    
        # all correct answer indices 
        all_idx = np.concatenate([count1_idx, count2_idx])
        correct_idx = np.sort(all_idx)
        
        all_time_dif = [] # empty list to store all time differences
    
        # loop over all correct answers
        for idx in correct_idx:
            
            # time when player press the button
            time_button = pd.to_datetime(level_data.loc[idx, 'timestamp'])
            # time when the character was shown
            time_character = pd.to_datetime(level_data.loc[idx - 1, 'timestamp'])
            
            time_dif       = time_button - time_character # time difference
            time_dif_float = time_dif.total_seconds() # convert is to seconds and float value
            all_time_dif.append(time_dif_float) # add the time difference to the list
    
        
        avg_time = np.mean(all_time_dif) # calculate the average time 
        time_matrix.append(avg_time) # store average time for each level in the list
        
    
    # store the info obtained from all files
    if len(all_score) == 0:  # check if all_score is empty
        all_score = score_matrix
    else:
        all_score = np.vstack((all_score, score_matrix)) 
        
    if len(all_time) == 0:   # check if all_time is empty
        all_time = time_matrix
    else:
        all_time = np.vstack((all_time, time_matrix))
        

# Plot data related info
#---------------------------

# Create DataFrames for scores and time
score_df = pd.DataFrame(all_score, columns=['Level 1', 'Level 2', 'Level 3'])
time_df = pd.DataFrame(all_time, columns=['Level 1', 'Level 2', 'Level 3'])

# Create a figure and axis
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Violin Plot for Score Distribution Across Levels
sns.violinplot(data=score_df, ax=axes[0])
axes[0].set_title('Score Distribution Across Levels')
axes[0].set_ylabel('Score')
axes[0].grid(True)

# Violin Plot for Time Distribution Across Levels
sns.violinplot(data=time_df, ax=axes[1])
axes[1].set_title('Time Distribution Across Levels')
axes[1].set_ylabel('Time (seconds)')
axes[1].grid(True)

plt.tight_layout()
plt.show()

plot1_name = 'Metadata_Visualization.png'
conf_file_path = os.path.join(main_folder, plot1_name)
plt.savefig(conf_file_path)

# Analyze 11,12, 18, 19 and 15 
#----------------------------------

other_file_names = ['data_5.parquet','data_11.parquet','data_12.parquet','data_18.parquet','data_19.parquet']

all_score_rest = [] # empty list to store avg score of other players
all_time_rest  = [] # empty list to store avg time of other players

# loop over them
for file_name in other_file_names:

    game_data = pd.read_parquet(os.path.join(data_folder, file_name))

    # Calculate the total correct answer in all levels
    count1_idx = game_data[game_data['event_id'] == 1].index # find event 1 indicies
    count1 = len(count1_idx) # count
    count2_idx = game_data[game_data['event_id'] == 2].index # find event 2 indicies
    count2 = len(count2_idx) # count

    # Calculate the total number of games
    count6_idx = game_data[game_data['event_id'] == 6].index
    n_games = len(count6_idx)
    
    score_total  = count1 + count2 # calculate the score
    
    # normalize the score (score relative to 25 games)
    norm_score = (25 * score_total) / n_games

    # Calculate the total time taken to answer in each level
    all_idx = np.concatenate([count1_idx, count2_idx])
    correct_idx = np.sort(all_idx)

    all_time_dif = [] # empty matrix to store all time differences
    
    # loop over all correct answers
    for idx in correct_idx:
            
        # time when player press the button
        time_button = pd.to_datetime(game_data.loc[idx, 'timestamp'])
        # time when the character was shown
        time_character = pd.to_datetime(game_data.loc[idx - 1, 'timestamp'])
            
        time_dif     = time_button - time_character # time difference
        time_dif_float = time_dif.total_seconds() # convert is to seconds and float value
        all_time_dif.append(time_dif_float)

    avg_time = np.mean(all_time_dif)  # calculate the average time 
    all_time_rest.append(avg_time) # store the average time values
    
    all_score_rest.append(norm_score) # store the scores


# convert the lists to arrays
all_score_rest = np.array(all_score_rest)
all_time_rest = np.array(all_time_rest)


## STEP 02: FEATURE ENGINEERING

# Until now 6 feature:
# 1. Level 1 - Score
# 2. Level 2 - Score
# 3. Level 3 - Score
# 4. Level 1 - Time
# 5. Level 2 - Time
# 6. Level 3 - Time

# Combining score feature into one feature
#-------------------------------------------

# Separate the levels
l1_score = all_score[:, 0]
l2_score = all_score[:, 1]
l3_score = all_score[:, 2]

# weights for levels based on difficulty
weights_score = np.array([0.1, 0.3, 0.6])

# calculate the weighted score
weighted_score = (weights_score[0] * l1_score +
                  weights_score[1] * l2_score +
                  weights_score[2] * l3_score)

# add the rest of the values from other players
all_weighted_score = np.concatenate((weighted_score, all_score_rest))


# Combining time feature into one feature
#-------------------------------------------

# Replace NaN values with the maximum time taken in the dataset
max_time = np.nanmax(all_time)
all_time[np.isnan(all_time)] = max_time

# Separate the levels
l1_time = all_time[:, 0]
l2_time = all_time[:, 1]
l3_time = all_time[:, 2]

# weights for levels based on difficulty considering as time increase performance decrease
weights_time = np.array([0.6, 0.3, 0.1])

# calculate the weighted score
weighted_time = (weights_time[0] * l1_time +
                 weights_time[1] * l2_time +
                 weights_time[2] * l3_time)

# add the other values 
all_weighted_time = np.concatenate((weighted_time, all_time_rest))


# Create the feature matrix
#-------------------------------
feature_matrix = np.column_stack((all_weighted_score, all_weighted_time))
# column 1: score
# column 2: time
# rows: feature vector of each player


# Normalize the values
#--------------------------------------

scaler = StandardScaler()
feature_matrix = scaler.fit_transform(feature_matrix)

weight_score = 0.6  # higher weight for score
weight_time  = 0.4   # lower weight for time

# Apply feature weighting
feature_matrix[:, 0] *= weight_score  # multiply score feature by its weight
feature_matrix[:, 1] *= weight_time   # multiply time feature by its weight


## STEP 03: CLUSTERING

# K-means clustering

# number of clusters 
k = 3;

# initialize k-means, set the random_state to get the same result each run
kmeans = KMeans(n_clusters=k, random_state=39)

# fit the model
kmeans.fit(feature_matrix)

# get the cluster labels for each data point
cluster_labels = kmeans.labels_

# plot the clustered data
plt.figure(figsize=(16, 12))
plt.scatter(feature_matrix[:, 0], feature_matrix[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5, s=80)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', s=400, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Score')
plt.ylabel('Time')
plt.legend()
plt.show()

plot2_name = 'Kmeans_Clustering.png'
conf_file_path = os.path.join(main_folder, plot2_name)
plt.savefig(conf_file_path)
