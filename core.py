# Import packages
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the example dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# exploratory data analysis
age_index = 0
sex_index = 1
bmi_index = 2
plt.scatter(diabetes_X[:, bmi_index], diabetes_y)
plt.xlabel('BMI (Normalized)')
plt.ylabel('Diabetes Expression Level')
plt.title('Diabetes Expression vs BMI')

#age
plt.scatter(diabetes_X[:, age_index], diabetes_y)
plt.xlabel('Age (Normalized)')
plt.ylabel('Diabetes Expression Level')
plt.title('Diabetes Expression vs Age')


# Use BMI index as the sole attribute
X = diabetes_X[:, bmi_index]

# Use a half of the dataset for test
test_size = X.shape[0] // 2

# Split the data into train & test
x_train = X[:-test_size]
x_test = X[-test_size:]

# Split the targets into training/testing sets
y_train = diabetes_y[:-test_size]
y_test = diabetes_y[-test_size:]


#find b and a to minimize loss
b = np.mean(y_train)
a = np.sum(x_train * y_train) / np.sum(x_train * x_train)

#calculate mse
mseTrain = np.sum((a * x_train + b - y_train) ** 2) / len(x_train)
mseTest = np.sum((a * x_test + b - y_test) ** 2) / len(x_test)

print("a = %.4f" % a)
print("b = %.4f" % b)
print("train mse = %.4f" % mseTrain)
print("test mse = %.4f" % mseTest)

# Plot results
fig, ax = plt.subplots()
ax.scatter(y_train, a * x_train + b, facecolor='blue', edgecolor='black')
ax.scatter(y_test, a * x_test + b, facecolor='red', edgecolor='black')
xRange = [0, 400]
yLine = xRange
ax.plot(xRange, yLine, color='black')
ax.set_xlabel('y')
ax.set_ylabel("predicted y")

#(see https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fgenerated%2Fsklearn.linear_model.LinearRegression.html)
regr = linear_model.LinearRegression()

# Split all the diabetes data into train & test
diabetes_X_train = diabetes_X[:-test_size]
diabetes_X_test = diabetes_X[-test_size:]

regr.fit(diabetes_X_train, y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test , diabetes_y_pred))


# Reuse previous code to plot
fig, ax = plt.subplots()
ax.scatter(y_train, regr.predict(diabetes_X_train), facecolor='blue', edgecolor='black')
ax.scatter(y_test, diabetes_y_pred, facecolor='red', edgecolor='black')
xRange = [0, 400]
yLine = xRange
ax.plot(xRange, yLine, color='black')
ax.set_xlabel('y')
ax.set_ylabel("predicted y")







