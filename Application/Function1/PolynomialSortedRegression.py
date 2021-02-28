import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import levene, bartlett, iqr
import joblib

#   constants for finding required indexes in the original data table
COMPLEX_AGE_INDEX = 2
TOTAL_ROOMS_INDEX = 3
TOTAL_BEDROOMS_INDEX = 4
COMPLEX_INHABITANTS_INDEX = 5
APARTMENTS_NR_INDEX = 6
MEDIAN_COMPLEX_VALUE_INDEX = 8

#   init empty arrays for taking data from file
data = []
answers = []
primary_data_from_file = []

#   read the data from file considering user's choice about how to make read (all the data or conform requirements)
with open("apartment_db.txt") as file_db:
    for line in file_db:
        row = line.split(",")
        current_row_usable_data = [
            float(row[COMPLEX_AGE_INDEX]),
            float(row[TOTAL_ROOMS_INDEX]),
            float(row[TOTAL_BEDROOMS_INDEX]),
            float(row[COMPLEX_INHABITANTS_INDEX]),
            float(row[APARTMENTS_NR_INDEX]),
            float(row[MEDIAN_COMPLEX_VALUE_INDEX])
        ]
        primary_data_from_file.append(current_row_usable_data)

#   get the data from primary taken data after optimizations conform user's choice made previously
for i in range(len(primary_data_from_file)):
    current_record = [
        primary_data_from_file[i][0],
        primary_data_from_file[i][1],
        primary_data_from_file[i][2],
        primary_data_from_file[i][3],
        primary_data_from_file[i][4],
    ]
    data.append(current_record)
    answers.append(primary_data_from_file[i][5])

#   transform standard arrays into numpy arrays
data, answers = np.array(data), np.array(answers)

#   fit data with answers to the LinearRegression module for analysis
model = LinearRegression().fit(data, answers)

#   try to analyze data and get scores
r_sq = model.score(data, answers)

#   show variables that are inner part of linear regression module
print("\n\t\tcoefficient of determination (R_square): " + str(r_sq))

#   generate possible answers using the same data as was provided to compare solutions
answers_predicted = model.predict(data)

print("\t\tLevine's test result = " + str(levene(answers, answers_predicted)[1]))
print("\t\tBartlett's test result = " + str(bartlett(answers, answers_predicted)[1]))

representable_plot_size = len(data)

#   calculate medium error
medium_error = 0
for i in range(representable_plot_size):
    medium_error += abs(answers[i] - answers_predicted[i])
medium_error /= representable_plot_size

print("\t\tMedium error is " + str(medium_error))

print("lwelkjfwljerg " + str(model.predict([[52.0, 1467.0, 190.0, 496.0, 177.0]])))

joblib.dump(model, 'polynomialRegressor.joblib')