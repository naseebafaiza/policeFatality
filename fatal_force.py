# Naseeba Faiza, 113498789

import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# CSV FILES
education =  pd.read_csv('education.csv')
income = pd.read_csv('income.csv')
poverty = pd.read_csv('poverty.csv')
race = pd.read_csv('share_race_by_city.csv')
police_killing_test = pd.read_csv('police_killings_test.csv')
police_killing_train = pd.read_csv('police_killings_train.csv')

# Cleaning up data
police_killing_test.dropna(inplace=True)
police_killing_train.dropna(inplace=True)

# Converting non-numerical values to NaN in race DataFrame
race['share_white'] = pd.to_numeric(race['share_white'], errors='coerce')
race['share_black'] = pd.to_numeric(race['share_black'], errors='coerce')
race['share_native_american'] = pd.to_numeric(race['share_native_american'], errors='coerce')
race['share_asian'] = pd.to_numeric(race['share_asian'], errors='coerce')
race['share_hispanic'] = pd.to_numeric(race['share_hispanic'], errors='coerce')

race.dropna(inplace=True)

# Combined share_race_by_city.csv, poverty.csv, education.csv, and income.csv by city names. 
# The merged data set contains all the information for each city in one data set.


merged_data = pd.merge(education, income, on=['Geographic Area', 'City'])
merged_data = pd.merge(merged_data, poverty, on=['Geographic Area', 'City'])
race = race.rename(columns={'Geographic area': 'Geographic Area'})
merged_data = pd.merge(merged_data, race, on=['Geographic Area', 'City'])

# Which state has the most fatal police shootings? Which city is the most dangerous?
# Calculate and visualize the states with the most number of fatal police shootings 
# by using data from police_killing_train.csv and police_killing_test.csv

state_counts_test = police_killing_test['state'].value_counts()
state_counts_train = police_killing_train['state'].value_counts()

state_counts = pd.concat([state_counts_test, state_counts_train], axis=0)
state_counts = state_counts.groupby(state_counts.index).sum()

plt.figure(figsize=(12, 6))
state_counts.plot(kind='bar')
plt.title('Fatal Police Shootings by State')
plt.xlabel('State')
plt.ylabel('Number of Fatal Shootings')
plt.show()

most_fatal_state = state_counts.idxmax()
most_fatal_shootings = state_counts[most_fatal_state]
print(f"The state with the most fatal police shootings is {most_fatal_state} with {most_fatal_shootings} fatal shootings.")

# Which state has the most fatal police shootings? Which city is the most dangerous?

city_counts_test = police_killing_test['city'].value_counts()
city_counts_train = police_killing_train['city'].value_counts()

city_counts = pd.concat([city_counts_test, city_counts_train], axis=0)
city_counts = city_counts.groupby(city_counts.index).sum()

city_counts = city_counts.sort_values(ascending=False)
print("Top 10 most 'dangerous' cities according to the number of fatalities:")
print(city_counts.head(10))

plt.figure(figsize=(12, 6))
city_counts.head(10).plot(kind='bar')
plt.title('Fatal Police Shootings by City')
plt.xlabel('City')
plt.ylabel('Number of Fatal Shootings')

plt.show()


frames = [police_killing_test, police_killing_train]
killings = pd.concat(frames, ignore_index=True)

plt.figure(figsize=(10, 6))
colors = {'B': 'blue', 'W': 'orange', 'H': 'green', 'A': 'red', 'N': 'purple', 'O': 'brown'}
labels = {'B': 'Black', 'W': 'White', 'H': 'Hispanic', 'A': 'Asian', 'N': 'Native American', 'O': 'Other'}

for race in colors:
    race_killings = killings[killings["race"] == race]
    ages = race_killings["age"].value_counts().sort_index()
    plt.plot(ages.index, ages.values, label=labels[race], color=colors[race])

plt.xlabel("Age")
plt.ylabel('Number of Fatal Shootings')
plt.title("Age Distribution of Shooting Victims by Race")
plt.legend()
plt.show()

print("The most common race fatally shot by police is white")

# Merging police killing test and training data
police_killing_test_reduced = police_killing_test[['state', 'city', 'race']]
police_killing_train_reduce = police_killing_train[['state','city','race']]
merged_police_killing = pd.concat([police_killing_test_reduced, police_killing_train_reduce])

# Organize and categorize the killings by race
grouped_total = merged_police_killing['race'].value_counts()
grouped_percentages = merged_police_killing['race'].value_counts(normalize=True) * 100

# Plotting total number of people killed per race
plt.figure(figsize=(10, 6))
grouped_total.plot(kind='bar')
plt.title('Total Number of People Killed per Race')
plt.xlabel('Race')
plt.ylabel('Number of Fatal Shootings')
plt.show()

# Plotting the proportion of people killed per race
plt.figure(figsize=(10, 6))
grouped_percentages.plot(kind='bar')
plt.title('Proportion of People Killed per Race')
plt.xlabel('Race')
plt.ylabel('Percentage')
plt.show()

print("Total Number of People Killed per Race:")
print(grouped_total)
print("\nProportion of People Killed per Race:")
print(grouped_percentages)

#ML

# Apply label encoding to all categorical columns in both datasets so they can be understood by our models
for col in police_killing_train.columns:
    if police_killing_train[col].dtype == 'object':
        le = LabelEncoder()
        police_killing_train[col] = le.fit_transform(police_killing_train[col])
        
for col in police_killing_test.columns:
    if police_killing_test[col].dtype == 'object':
        le = LabelEncoder()
        police_killing_test[col] = le.fit_transform(police_killing_test[col])

print(police_killing_train.head())

# Prepare the data by separating the features and the target variable
# Drop the 'race' column from the 'police_killings_train' dataset to create the feature matrix 'X_train'
# Set the 'race' column as the target variable 'y_train'
X_train = police_killing_train.drop(columns=['race'])
y_train = police_killing_train['race']

# Drop the 'race' column from the 'police_killings_test' dataset to create the feature matrix 'X_test'
X_test = police_killing_test.drop(columns=['race'])

# K-Nearest Neighbors (KNN) Algorithm
# Create a KNeighborsClassifier object with n_neighbors=5 and fit it to the training data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test data using knn.predict() and assign the results to y_pred
y_pred_knn = knn.predict(X_test)

# Neural Networks Algorithm
# Create an MLPClassifier object with one hidden layer of 100 neurons and fit it to the training data
mlp = MLPClassifier(hidden_layer_sizes=(100,))
mlp.fit(X_train, y_train)

# Make predictions on the test data using mlp.predict() and assign the results to y_pred
y_pred_mlp = mlp.predict(X_test)

# Linear Regression Algorithm
# Create a LinearRegression object and fit it to the training data
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions on the test data using linear_reg.predict() and assign the results to y_pred
y_pred_lin = linear_reg.predict(X_test)

# Find the accuracy of the models:

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# Inverse transform the label encoded target variable
le = LabelEncoder()
le.fit(police_killing_train['race'])
y_test = le.inverse_transform(police_killing_test['race'])

# Calculate the accuracy scores for K-Nearest Neighbors (KNN) and Neural Networks models
acc_knn = accuracy_score(y_test, y_pred_knn)
print('Accuracy of K-Nearest Neighbors (KNN) model:', acc_knn)

acc_mlp = accuracy_score(y_test, y_pred_mlp)
print('Accuracy of Neural Networks model:', acc_mlp)

# Calculate the R-squared value for the Linear Regression model
r2 = r2_score(y_test, y_pred_lin)
print("R-squared value of Linear Regression model:", r2)

# Based on the given accuracies and R-squared value, it can be concluded that neither the KNN nor 
# Neural Networks models are effective in predicting the race given police killings data. The KNN 
# accuracy indicates that the model performs only slightly better than random guessing.
