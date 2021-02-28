import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import levene, bartlett, iqr
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt

#   constants for finding required indexes in the original data table
COMPLEX_AGE_INDEX = 2
TOTAL_ROOMS_INDEX = 3
TOTAL_BEDROOMS_INDEX = 4
COMPLEX_INHABITANTS_INDEX = 5
APARTMENTS_NR_INDEX = 6
MEDIAN_COMPLEX_VALUE_INDEX = 8

#   optimization strategy choice constants
STANDARDIZATION_METHOD_INDEX = 1
GAUSSIAN_DISTRIBUTION_METHOD_INDEX = 2
QUANTILE_TRANSFORMATION_METHOD_INDEX = 3
NORMALIZATION_OF_INPUTS_METHOD_INDEX = 4

#   init empty arrays for taking data from file
data = []
answers = []
primary_data_from_file = []

#   ask about how to perform data reading from original file
print(
    "Do you want to read all the data or only the part of the lab? Available answers:\n" +
    "\ty - yes, read all the data\n"
    "\tn - no, read only lab's part of the data"
)
choice_about_analyzable_data = input(">>>\t")

#   read the data from file considering user's choice about how to make read (all the data or conform lab requirements)
with open("apartment_db.txt") as file_db:
    if choice_about_analyzable_data == 'n':
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

    elif choice_about_analyzable_data == 'y':
        for line in file_db:
            row = line.split(",")
            current_row_usable_data = [
                float(row[0]),
                float(row[1]),
                float(row[COMPLEX_AGE_INDEX]),
                float(row[TOTAL_ROOMS_INDEX]),
                float(row[TOTAL_BEDROOMS_INDEX]),
                float(row[COMPLEX_INHABITANTS_INDEX]),
                float(row[APARTMENTS_NR_INDEX]),
                float(row[7]),
                float(row[MEDIAN_COMPLEX_VALUE_INDEX])
            ]
            primary_data_from_file.append(current_row_usable_data)

print(
    "What data optimization method do you want to use? Available answers:\n"
    "\t1 - Standardization;\n" +
    "\t2 - Gaussian distribution;\n" +
    "\t3 - Quantile transformer;\n" +
    "\t4 - Normalization of the inputs;\n"
    "\tAny other number - no optimization required"
)
method_for_data_optimization = input(">>>\t")

#   get the data from primary taken data after optimizations conform user's choice made previously
if choice_about_analyzable_data == "y":
    for i in range(len(primary_data_from_file)):
        current_record = [
            primary_data_from_file[i][0], primary_data_from_file[i][1],
            primary_data_from_file[i][COMPLEX_AGE_INDEX],
            primary_data_from_file[i][TOTAL_ROOMS_INDEX],
            primary_data_from_file[i][TOTAL_BEDROOMS_INDEX],
            primary_data_from_file[i][COMPLEX_INHABITANTS_INDEX],
            primary_data_from_file[i][APARTMENTS_NR_INDEX],
            primary_data_from_file[i][7]
        ]
        data.append(current_record)
        answers.append(primary_data_from_file[i][MEDIAN_COMPLEX_VALUE_INDEX])

elif choice_about_analyzable_data == "n":
    for i in range(len(primary_data_from_file)):
        current_record = [
            primary_data_from_file[i][0],
            primary_data_from_file[i][1],
            primary_data_from_file[i][2],
            primary_data_from_file[i][3],
            primary_data_from_file[i][4]
        ]
        data.append(current_record)
        answers.append(primary_data_from_file[i][5])

#   apply chosen optimization for the data (considering small amount of options "if-else" relation implemented
transformer = None

if int(method_for_data_optimization) == STANDARDIZATION_METHOD_INDEX:
    transformer = preprocessing.StandardScaler()
    data = transformer.fit_transform(data)

elif int(method_for_data_optimization) == GAUSSIAN_DISTRIBUTION_METHOD_INDEX:
    transformer = preprocessing.PowerTransformer(method="box-cox", standardize=True)
    data = transformer.fit_transform(data)

elif int(method_for_data_optimization) == QUANTILE_TRANSFORMATION_METHOD_INDEX:
    transformer = preprocessing.QuantileTransformer(output_distribution="normal", random_state=0)
    data = transformer.fit_transform(data)

if int(method_for_data_optimization) == NORMALIZATION_OF_INPUTS_METHOD_INDEX:
    data = preprocessing.normalize(data)

#   choose if there is IQR outliers filtering required
choice_of_outliers_filtering = input("Do you want to perform Outliers IQR filtering (y - yes, n - no)?\n>>>\t")
if choice_of_outliers_filtering == "y":
    Q1 = np.quantile(answers, 0.25)
    Q3 = np.quantile(answers, 0.75)
    IQR = iqr(answers)

    non_outliers_indexes = []
    for i in range(len(answers)):
        if (Q1 - (1.5 * IQR)) < answers[i] < (Q3 + (1.5 * IQR)):
            non_outliers_indexes.append(i)

    clean_answers = []
    clean_data = []
    for index in non_outliers_indexes:
        clean_answers.append(answers[index])
        clean_data.append(data[index])

    answers = clean_answers
    data = clean_data


#   transform standard arrays into numpy arrays
data, answers = np.array(data), np.array(answers)

polynomial_degree = input(
    "Choose a regression polynomial degree (recommended use of 1-5 degree, otherwise long processing occurs):\t"
)

#   init polynomial features object
transformer = PolynomialFeatures(degree=int(polynomial_degree), include_bias=True)

#   fit input for polynomial analysis
transformer.fit(data)

#   polynomial variables values generation out of input
polynomial_data = transformer.transform(data)

#   fit data with answers to the LinearRegression module for analysis
model = LinearRegression().fit(polynomial_data, answers)

#   try to analyze data and get scores
r_sq = model.score(polynomial_data, answers)

#   show variables that are inner part of linear regression module
print("\n\t\tcoefficient of determination (R_square): " + str(r_sq))

#   generate possible answers using the same data as was provided to compare solutions
answers_predicted = model.predict(polynomial_data)

print("\t\tLevine's test result = " + str(levene(answers, answers_predicted)[1]))
print("\t\tBartlett's test result = " + str(bartlett(answers, answers_predicted)[1]))

representable_plot_size = len(data)

#   calculate medium error
medium_error = 0
for i in range(representable_plot_size):
    medium_error += abs(answers[i] - answers_predicted[i])
medium_error /= representable_plot_size

print("\t\tMedium error is " + str(medium_error))

#   set all original answers as red dots on the diagram
plt.scatter(range(0, representable_plot_size), answers[:representable_plot_size], color="red")
#   set predicted answers as blue plot on the diagram
plt.plot(range(0, representable_plot_size), answers_predicted[:representable_plot_size], color="blue")

plt.title("Graph for original answers and predictions (originals - red, predicted - blue)")
plt.xlabel("record ID")
plt.ylabel("Median Complex Value")
plt.grid(color="black")
plt.show()