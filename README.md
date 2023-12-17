# policeFatality
Data analysis and machine learning to study fatal police shootings in the U.S., using data from multiple sources to explore state and city-level patterns, race distributions, and applies machine learning models to predict racial demographics of victims.

This code performs various data analysis and machine learning tasks using pandas, matplotlib, and scikit-learn libraries. The necessary libraries 
are imported: pandas, matplotlib, IPython, scikit-learn modules. Several CSV files are loaded into pandas DataFrames: education, income, poverty, 
race, police_killing_test, and police_killing_train. Data cleaning is performed: NaN values are dropped from police_killing_test and police_killing_train 
DataFrames, and non-numerical values in the race DataFrame are converted to NaN. The DataFrames are merged using common columns (Geographic Area and City) 
to create the merged_data DataFrame. The code then determines the state and city with the most fatal police shootings by counting the occurrences in 
police_killing_test and police_killing_train DataFrames. The results are plotted using bar charts. The age distribution of shooting victims by race 
is visualized by plotting the number of fatal shootings against age for each race. The most common race fatally shot by police is identified.

The police killing test and training data are merged, and the killings are organized and categorized by race. The total number and proportion of people 
killed per race are plotted. Machine learning tasks are performed using scikit-learn:

a. Label encoding is applied to categorical columns in police_killing_train and police_killing_test DataFrames.

b. The data is prepared by separating features (X) and the target variable (y) from the train and test datasets.

c. The K-Nearest Neighbors (KNN), Neural Networks, and Linear Regression algorithms are applied to the data using scikit-learn's respective classes.

d. Predictions are made on the test data using the trained models.

e. Accuracy scores are calculated for the KNN, Neural Networks, and Linear Regression models.

Finally, the accuracy scores and R-squared value are printed, and a conclusion is drawn regarding the effectiveness of the models in predicting race 
given police killings data.
