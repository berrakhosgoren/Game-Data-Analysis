# Game Data Analysis

**Author:** Berrak Hoşgören

The objective of this project is to analyze game data and categorize players into three groups: bad performers, normal performers, and good performers. The project involves data processing, visualization, and k-means clustering.

## Game Description

In the game, players are presented with a sequence of individual letters on the screen, one at a time. The game consists of multiple levels:
- **Level 1**: Players must press the green button when the current letter matches the previous one.
- **Level 2**: Players press the green button if the current letter matches the one shown two positions earlier.
- **Level 3**: Players press the green button if the current letter matches the one shown three positions earlier.

## Analyzing the Data
The `main.py` script performs the following steps:

**Data Processing:**
- The script reads game data files and separates them into levels.
- It calculates scores based on correct and incorrect responses for each level.
- It calculates the time taken to respond to each character for each level

**Feature Engineering:**
- It combines scores and response times from each level into weighted features.

**Clustering:**
- It performs k-means clustering on the data to categorize players into three groups.
  
**Visualization:**
- It visualizes the distribution of scores and response times across levels using violin plots.
- It generates a visualization of k-means clustering results to categorize players based on performance.

