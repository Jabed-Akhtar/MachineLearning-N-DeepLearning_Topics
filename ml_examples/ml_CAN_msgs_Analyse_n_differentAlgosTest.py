# -*- coding: utf-8 -*-
"""
 * Note: not finished!!!
 ******************************************************************************
 * @author     : Jabed-Akhtar (github)
 * @Created on : Tue Apr  5 22:18:24 2022
 ******************************************************************************
 * @file       : ml_CAN_msgs_Analyse_n_differentAlgosTest.py
 * @brief      : an example of ---
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
 *      - used datasets and files can be found in folder: '/datasets/CAN_logData.csv'
 *      - trained model is saved at location: '/trainedModels/---.pickle'
 *      - plots of predictions can be found at location: '/docs_images_logs__/CAN_msgs_Analyse_n_differentAlgosTest_***.***'
 * 
 ******************************************************************************
"""


#Imports ======================================================================
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


#Variables ====================================================================
PATH_SCRIPT = os.path.dirname(__file__)
PATH_DATASET = os.path.join(PATH_SCRIPT, 'datasets/CAN_logData.csv')

predict = 'timestamp'




#Reading data =================================================================
data = pd.read_csv(PATH_DATASET)
timestamp_data = data['timestamp']
print(data['dlc'].dtypes)
msg_list = []

for n in range(2):
    dat_print = data.iloc[n]
    #msg_list = [dat_print.timestamp, hex(int(dat_print.id, 16)), dat_print.dlc, dat_print.data]
    msg_list = [dat_print.timestamp, hex(0x123), dat_print.dlc, dat_print.data]
    print(hex(int(dat_print.id, 16)))
    print(msg_list)



# Allocating 10% of data for testing
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


#Building the model and fitting model to data =================================
# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
# acc = model.score(X_test, Y_test)
# print(acc)


#Testings/Predictions =========================================================
# predictions1 = model.predict([ [21, 1], [22, 0] ])
# predictions2 = model.predict(X_test)
# score = accuracy_score(Y_test, predictions2)


#Saving model =================================================================
# joblib.dump(model, 'decisionTreeClassifier_music.joblib')

    
#Loading/Using saved model with joblib
# model_loaded = joblib.load('decisionTreeClassifier_music.joblib')
# # testing with loaded model
# predictions_withLoadedModel = model.predict([ [21, 1], [22, 0] ])
            
            
#Plotting/Visualizing =========================================================
# tree.export_graphviz(model, out_file='DecisionTreeClassifier_Music-Recommender.dot',
#                      feature_names=['age', 'gender'],
#                      class_names=sorted(Y_train.unique()),
#                      label='all',
#                      rounded=True,
#                      filled=True)


# ****************************** END OF FILE **********************************