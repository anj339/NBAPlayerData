# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:32:59 2024

@author: anjou
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import itertools


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import statsmodels.api as sm

player_df = pd.read_csv('PlayerData.csv')
#lets start cleaning up and extracting information.
unique_players = player_df.Name.nunique()
unique_injuries = player_df.Injuries.nunique()

print()
print()
print("Heat Database")
Miami_Heat_df = pd.DataFrame(player_df)

# Clean Salary column
Miami_Heat_df['Salary'] = Miami_Heat_df['Salary'].replace('[\$,]', '', regex=True).astype(float)

# Fill missing injuries with 'No Injury'
Miami_Heat_df['Injuries'] = Miami_Heat_df['Injuries'].replace('', 'No Injury')

# Display the cleaned DataFrame
print(Miami_Heat_df)


print('There are {} players in the dataset.'.format(unique_players))

print('There are {} injuries in the dataset.'.format(unique_injuries))

# One-hot encode the 'Injuries' column
Miami_Heat_df_encoded = pd.get_dummies(Miami_Heat_df, columns=['Injuries'], drop_first=True)

# Display the encoded DataFrame
print(Miami_Heat_df_encoded)

# Features and target
X = Miami_Heat_df_encoded.drop(columns=['Rk', 'Name', 'Salary'])
y = Miami_Heat_df_encoded['Salary']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = sm.OLS(y_train, X_train).fit()

# Display the model summary
print(model.summary())

# Predict future salaries
y_pred = model.predict(X_test)

# Ensure non-negative predictions
y_pred = y_pred.apply(lambda x: max(x, 0))

# Display predictions
predictions = pd.DataFrame({'Name': Miami_Heat_df.loc[X_test.index, 'Name'], 'Predicted_Salary': y_pred})
print(predictions)



#Player Data from NBA Playoff Season 23-24'
# Google Sheets URL
sheet_name = 'NBA_DATA' # replace with your own sheet name
sheet_id = '14osUVbDxJhuCplG-LqVbsjXOR9upVl_MWPJDXgsrn9k' # replace with your sheet's ID
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# Load the data into a DataFrame
df = pd.read_csv(url)

# Display the DataFrame
print(df)

print(df.head())


#determining the relationship between age and wins

import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go

# Renaming columns for easier access
df.columns = ['Player', 'Team', 'Age', 'Games_Played', 'Wins', 'Loses', 'Minutes_Per_Game', 'Points', 'Field_Goals_Made', 'Field_Goals_Attempted', 'Field_Goal_Percentage', 'Three_Pointers_Made', 'Three_Pointers_Attempted', 'Three_Point_Percentage', 'Free_Throws_Made', 'Free_Throws_Attempted', 'Free_Throw_Percentage', 'Offensive_Rebounds', 'Defensive_Rebounds', 'Total_Rebounds', 'Assists', 'Turnovers', 'Steals', 'Blocks',  'Personal_Fouls', 'Fantasay Points', 'Double_Points', 'Triple_Doubles', 'Plus_Minus', 'Unamed']
print('updated databse')
print(df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
df = pd.read_csv('https://docs.google.com/spreadsheets/d/14osUVbDxJhuCplG-LqVbsjXOR9upVl_MWPJDXgsrn9k/gviz/tq?tqx=out:csv&sheet=NBA_DATA')

# Renaming columns for easier access
df.columns = ['Player', 'Team', 'Age', 'Games_Played', 'Minutes_Per_Game', 'Field_Goals_Made', 'Field_Goals_Attempted', 'Field_Goal_Percentage', 'Three_Pointers_Made', 'Three_Pointers_Attempted', 'Three_Point_Percentage', 'Free_Throws_Made', 'Free_Throws_Attempted', 'Free_Throw_Percentage', 'Offensive_Rebounds', 'Defensive_Rebounds', 'Total_Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Personal_Fouls', 'Points', 'Unnamed_23', 'Unnamed_24', 'Unnamed_25', 'Unnamed_26', 'Unnamed_27', 'Unnamed_28', 'Unnamed_29']

# Drop unnecessary columns
df = df.drop(columns=['Unnamed_23', 'Unnamed_24', 'Unnamed_25', 'Unnamed_26', 'Unnamed_27', 'Unnamed_28', 'Unnamed_29'])

# Fill missing values
df = df.fillna(0)

# 1. Predicting Player Performance
# Features and target variables
X = df[['Age', 'Minutes_Per_Game', 'Games_Played']]

# Predicting Points
y_points = df['Points']
X_train, X_test, y_train, y_test = train_test_split(X, y_points, test_size=0.2, random_state=42)
model_points = LinearRegression()
model_points.fit(X_train, y_train)
pred_points = model_points.predict(X_test)
mse_points = mean_squared_error(y_test, pred_points)

# Predicting Assists
y_assists = df['Assists']
X_train, X_test, y_train, y_test = train_test_split(X, y_assists, test_size=0.2, random_state=42)
model_assists = LinearRegression()
model_assists.fit(X_train, y_train)
pred_assists = model_assists.predict(X_test)
mse_assists = mean_squared_error(y_test, pred_assists)

# Predicting Rebounds
y_rebounds = df['Total_Rebounds']
X_train, X_test, y_train, y_test = train_test_split(X, y_rebounds, test_size=0.2, random_state=42)
model_rebounds = LinearRegression()
model_rebounds.fit(X_train, y_train)
pred_rebounds = model_rebounds.predict(X_test)
mse_rebounds = mean_squared_error(y_test, pred_rebounds)

# 2. Team Performance
team_stats = df.groupby('Team').mean(numeric_only=True)
X_team = team_stats[['Age', 'Minutes_Per_Game', 'Games_Played']]
y_team_points = team_stats['Points']
X_train, X_test, y_train, y_test = train_test_split(X_team, y_team_points, test_size=0.2, random_state=42)
model_team_points = LinearRegression()
model_team_points.fit(X_train, y_train)
pred_team_points = model_team_points.predict(X_test)
mse_team_points = mean_squared_error(y_test, pred_team_points)

# 3. Player Longevity
# Define a long career as playing more than 10 years
# Assuming 'Games_Played' is a proxy for career length
long_career = df['Games_Played'] > 10
y_longevity = long_career.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_longevity, test_size=0.2, random_state=42)
model_longevity = RandomForestClassifier(random_state=42)
model_longevity.fit(X_train, y_train)
pred_longevity = model_longevity.predict(X_test)
accuracy_longevity = accuracy_score(y_test, pred_longevity)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('https://docs.google.com/spreadsheets/d/14osUVbDxJhuCplG-LqVbsjXOR9upVl_MWPJDXgsrn9k/gviz/tq?tqx=out:csv&sheet=NBA_DATA')

# Renaming columns for easier access
df.columns = ['Player', 'Team', 'Age', 'Games_Played', 'Minutes_Per_Game', 'Field_Goals_Made', 'Field_Goals_Attempted', 'Field_Goal_Percentage', 'Three_Pointers_Made', 'Three_Pointers_Attempted', 'Three_Point_Percentage', 'Free_Throws_Made', 'Free_Throws_Attempted', 'Free_Throw_Percentage', 'Offensive_Rebounds', 'Defensive_Rebounds', 'Total_Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Personal_Fouls', 'Points', 'Unnamed_23', 'Unnamed_24', 'Unnamed_25', 'Unnamed_26', 'Unnamed_27', 'Unnamed_28', 'Unnamed_29']

# Drop unnecessary columns
df = df.drop(columns=['Unnamed_23', 'Unnamed_24', 'Unnamed_25', 'Unnamed_26', 'Unnamed_27', 'Unnamed_28', 'Unnamed_29'])

# Fill missing values
df = df.fillna(0)

# Define the target variable for predicting participation in the 2025 NBA season
# Assuming players who have played more than 15 games in the current season are likely to participate in 2025
participate_2025 = df['Games_Played'] > 15
y_participate_2025 = participate_2025.astype(int)
print(participate_2025)
print(y_participate_2025)

# Features
X = df[['Age', 'Minutes_Per_Game', 'Games_Played']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_participate_2025, test_size=0.2, random_state=42)

# Build the Random Forest Classifier model
model_participate_2025 = RandomForestClassifier(random_state=42)
model_participate_2025.fit(X_train, y_train)

# Make predictions
pred_participate_2025 = model_participate_2025.predict(X_test)

# Calculate accuracy
accuracy_participate_2025 = accuracy_score(y_test, pred_participate_2025)


print("Accuracy of player preductions")
print(accuracy_participate_2025)

# Define the target variable for predicting participation in the 2025 NBA season
# Assuming players who have played more than 15 games in the current season are likely to participate in 2025
participate_2025 = df['Games_Played'] > 15
y_participate_2025 = participate_2025.astype(int)

# Features
X = df[['Age', 'Minutes_Per_Game', 'Games_Played']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_participate_2025, test_size=0.2, random_state=42)

# Build the Random Forest Classifier model
model_participate_2025 = RandomForestClassifier(random_state=42)
model_participate_2025.fit(X_train, y_train)

# Make predictions
pred_participate_2025 = model_participate_2025.predict(X_test)

# Calculate accuracy
accuracy_participate_2025 = accuracy_score(y_test, pred_participate_2025)

# Create a new column in the DataFrame for participation prediction
participation_prediction = model_participate_2025.predict(X)
df['Participate_2025'] = ['Yes' if pred == 1 else 'No' for pred in participation_prediction]

# Display the results
print("Accuracy of player participation predictions:", accuracy_participate_2025)
print(df[['Player', 'Participate_2025']].head())

# Count the number of players who will and will not participate in the 2025 NBA season
participation_counts = df['Participate_2025'].value_counts()

# Separate tables for players who will and will not participate
will_participate = df[df['Participate_2025'] == 'Yes'][['Player', 'Participate_2025']]
will_not_participate = df[df['Participate_2025'] == 'No'][['Player', 'Participate_2025']]

print("Participation counts:\n", participation_counts)
print("Players who will participate:\n", will_participate)
print("Players who will not participate:\n", will_not_participate)


#Model to predict with players will transfer to a new team based on performance metrics
# Define the target variable for predicting player transfer
# Assuming players who have played fewer games in the playoffs are more likely to transfer
transfer = df['Games_Played'] < 15
y_transfer = transfer.astype(int)

# Features
X = df[['Games_Played', 'Points', 'Personal_Fouls', 'Field_Goals_Made', 'Field_Goals_Attempted', 'Three_Pointers_Made', 'Three_Pointers_Attempted', 'Free_Throws_Made', 'Free_Throws_Attempted', 'Total_Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_transfer, test_size=0.2, random_state=42)

# Build the Random Forest Classifier model
model_transfer = RandomForestClassifier(random_state=42)
model_transfer.fit(X_train, y_train)

# Make predictions
pred_transfer = model_transfer.predict(X_test)

# Calculate accuracy
accuracy_transfer = accuracy_score(y_test, pred_transfer)

# Create a new column in the DataFrame for transfer prediction
transfer_prediction = model_transfer.predict(X)
df['Transfer'] = ['Yes' if pred == 1 else 'No' for pred in transfer_prediction]

# Display the results
print("Accuracy of player transfer predictions:", accuracy_transfer)
print(df[['Player', 'Transfer']].head())

# Count the number of players who will and will not transfer
transfer_counts = df['Transfer'].value_counts()

# Separate tables for players who will and will not transfer
will_transfer = df[df['Transfer'] == 'Yes'][['Player', 'Transfer']]
will_not_transfer = df[df['Transfer'] == 'No'][['Player', 'Transfer']]

print("Transfer counts:\n", transfer_counts)
print("Players who will transfer:\n", will_transfer)
print("Players who will not transfer:\n", will_not_transfer)