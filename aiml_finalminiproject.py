# -*- coding: utf-8 -*-

# Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

# Load dataset (ensure data.csv is in the same folder)
data = pd.read_csv('data.csv')
print(f"Dataset successfully Imported of Shape : {data.shape}")

# Drop irrelevant columns
irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
data = data.drop(irrelevant, axis=1)

# Filter consistent teams and overs >= 5
const_teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]
data = data[(data['batting_team'].isin(const_teams)) & (data['bowling_team'].isin(const_teams))]
data = data[data['overs'] >= 5.0]

# Encode team names
le = LabelEncoder()
for col in ['batting_team', 'bowling_team']:
    data[col] = le.fit_transform(data[col])

# One-hot encode teams
columnTransformer = OneHotEncoder()
encoded_data = columnTransformer.fit_transform(data[['batting_team', 'bowling_team']]).toarray()

# Combine with numerical features
numerical_data = data.drop(['batting_team', 'bowling_team'], axis=1).values
final_data = np.concatenate((encoded_data, numerical_data), axis=1)

# Create feature labels
features = final_data[:, :-1]
labels = final_data[:, -1]

# Split the data
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train Random Forest Regressor (recommended model)
forest = RandomForestRegressor()
forest.fit(train_features, train_labels)

# Evaluate
print("---- Random Forest Evaluation ----")
print(f"Train Score: {forest.score(train_features, train_labels) * 100:.2f}%")
print(f"Test Score: {forest.score(test_features, test_labels) * 100:.2f}%")
y_pred = forest.predict(test_features)
print(f"MAE: {mean_absolute_error(test_labels, y_pred):.2f}")
print(f"MSE: {mean_squared_error(test_labels, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_labels, y_pred)):.2f}")

# Save the model
dump(forest, 'forest_model.pkl')

print("\n✅ Model saved as 'forest_model.pkl'")
