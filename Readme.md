
# NBA Player Analysis and Prediction

## Overview

Welcome to the NBA Player Analysis and Prediction project! This project aims to provide in-depth analysis and predictions related to Miami Heat players and other NBA players. Specifically, it includes:
Future Salary Prediction: Analyzing current and historical data to predict future salaries of Miami Heat players.
Playoff Participation Analysis: Evaluating playoff performance data to estimate each NBA player's likelihood of participating in the next season and their potential for transfer.

## Features
Salary Prediction:

Predict future salaries of Miami Heat players using historical performance metrics and salary data.

Playoff Data Analysis:

Analyze playoff data to assess the chances of NBA players participating in the next season.
Estimate the likelihood of player transfers based on playoff performance and other relevant factors.

## Project Steps
Load the Dataset: The dataset is loaded from a Google Sheets link and unnecessary columns are dropped.
Data Preprocessing: Missing values are filled, and columns are renamed for easier access.

Define Target Variable: The target variable is defined based on the assumption that players who played fewer games in the playoffs are more likely to transfer.

Feature Selection: Relevant features for the model are selected, including games played, points, fouls, and other performance metrics.

Data Splitting: The dataset is split into training and testing sets using an 80-20 split.

Model Training: A Random Forest Classifier model is trained on the training set.

Predictions and Evaluation: The model makes predictions on the testing set, and the accuracy of the model is calculated.

Transfer Prediction: The model is used to predict whether each player in the dataset will transfer, and the predictions are added as a new column in the DataFrame.

Results Display: The results, including the accuracy, transfer counts, and lists of players who will and will not transfer, are displayed.

## Results
Model Accuracy: The Random Forest Classifier model achieved an accuracy of 100% in predicting player transfers.

Transfer Predictions: The model predicted which players are likely to transfer to a new team based on their playoff performance metrics.

The residual analysis for the points prediction model reveals the following:

Residual Plot: The residuals are scattered randomly around zero, indicating that the model does not exhibit obvious patterns or biases and captures the underlying data structure well.

Histogram of Residuals: The residuals appear to follow a roughly normal distribution, suggesting that the model's errors are normally distributed.

Q-Q Plot: The residuals lie approximately along the reference line, further confirming that the residuals are normally distributed.
Overall, the residual analysis suggests that the points prediction model is well-fitted and meets the assumptions of linear regression.

## Conclusion
This project demonstrates the use of a Random Forest Classifier to predict NBA player transfers based on performance data. The high accuracy of the model suggests that it effectively captures the relationships between performance metrics and the likelihood of a player transferring to a new team.

