# -*- coding: utf-8 -*-
"""
 ******************************************************************************
 * @author     : Jabed-Akhtar (github)
 * @Created on : Fri Apr  1 03:18:48 2022
 ******************************************************************************
 * @file       : ml_DecisionTreeClassifier_Music.py
 * @brief      : an example of classifying music (music recommender)
 ******************************************************************************
 * :Steps      :
 *      1. Importing python libraries
 *      2. Defining variables
 *      3. Reading data
 *      4. Building the model and fitting model to data
 *      5. Testings/Predictions
 *      6. Saving model
 *      7. Visualizing the model trained with DecisionTreeClassified
 * :Descriptions/Infos:
 *      - a source used within this script: https://youtu.be/7eh4d6sabA0
 *      - used datasets and files can be found in folder: '/datasets/music.csv'
 *      - trained model is saved at location: '/trainedModels/---.pickle'
 *      - a dot file of visualization of the model can be found at location: '/docs_images_logs__/DecisionTreeClassifier_Music-Recommender.dot'
 *          -> visualization can be seen with VS-Code (Graphviz (dot) ... extension needs to be installed)
 *          -> a screenshot of the visualization in VS-Code can be found at location: '/docs_images_logs__/DecisionTreeClassifier_Music-Recommender-dotPreview_VSCode.jpg'
 * 
 ******************************************************************************
"""


#Imports ======================================================================
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
#from sklearn.externals import joblib
import matplotlib.pyplot as plt


#Variables ====================================================================
predict = "genre"


#Reading data =================================================================
data = pd.read_csv('datasets/music.csv')
X = data.drop(columns=[predict]) # dropping genre column, as it is the output
Y = data[predict]
print(Y)


# Allocating 10% of data for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


#Building the model and fitting model to data =================================
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
acc = model.score(X_test, Y_test)
print(acc)


#Testings/Predictions =========================================================
predictions1 = model.predict([ [21, 1], [22, 0] ])
predictions2 = model.predict(X_test)
score = accuracy_score(Y_test, predictions2)


#Saving model =================================================================
joblib.dump(model, 'decisionTreeClassifier_music.joblib')

    
#Loading/Using saved model with joblib
model_loaded = joblib.load('decisionTreeClassifier_music.joblib')
# testing with loaded model
predictions_withLoadedModel = model.predict([ [21, 1], [22, 0] ])
            
            
#Plotting/Visualizing =========================================================
tree.export_graphviz(model, out_file='DecisionTreeClassifier_Music-Recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(Y_train.unique()),
                     label='all',
                     rounded=True,
                     filled=True)


# ****************************** END OF FILE **********************************