# -*- coding: utf-8 -*-
"""
 ******************************************************************************
 * @author     : Jabed-Akhtar (github)
 * @Created on : Mon Mar 16 23:04:05 2022
 ******************************************************************************
 * @file       : ml_LinearRegression_StudentPerformance.py
 * @brief      : an example of predicting student grade using Linear-Regression method/algo
 ******************************************************************************
 * :Steps      :
 *      1. Importing python libraries
 *      2. Defining variables
 *      3. Reading data
 *      4. Building the model and fitting model to data
 *      5. Testings/Predictions
 *      6. Saving model
 *      7. Saving the best model - a method
 *      8. Plotting
 * :Descriptions/Infos:
 *      - a source used within this script: https://www.techwithtim.net/tutorials/machine-learning-python/
 *      - dataset can be found at: https://archive.ics.uci.edu/ml/datasets/student+performance
 *      - used datasets and files can be found in folder: '/datasets/student/student-mat.csv'
 *          -> please also read 'student.txt' file for more information about the here used dataset
 *      - trained model is saved at location: '/trainedModels/linearRegression_studentPerfor.pickle'
 *      - a plot of prediction can be found at location: '/docs_images/linearRegression_studentPerformace_predictionPlot_xxx.png'
 * 
 ******************************************************************************
"""


#Imports ======================================================================
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
from matplotlib import style


#Variables ====================================================================
predict = "G3"


#Reading data =================================================================
data = pd.read_csv("datasets/student/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


#Building the model and fitting model to data =================================
linear = linear_model.LinearRegression()
linear.fit(X_train, Y_train)
acc = linear.score(X_test, Y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)


#Testings/Predictions =========================================================
predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], Y_test[x])


#Saving model =================================================================
with open("linearRegression_studentPerfor.pickle", "wb") as f:
    pickle.dump(linear, f)
    
#Loading/Using saved model with pickle
pickle_in = open("linearRegression_studentPerfor.pickle", "rb")
linear = pickle.load(pickle_in)


#Saving the best model - a method =============================================
best = 0
for _ in range(30):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    
    linear = linear_model.LinearRegression()
    linear.fit(X_train, Y_train)
    acc = linear.score(X_test, Y_test)
    print(acc)
    
    if acc > best:
        best = acc
        with open("trainedModels/linearRegression_studentPerfor.pickle", "wb") as f:
            pickle.dump(linear, f)
            
            
#Plotting =====================================================================
p = "absences"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()


# ****************************** END OF FILE **********************************