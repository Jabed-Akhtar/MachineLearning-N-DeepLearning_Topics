# -*- coding: utf-8 -*-
"""
******************************************************************************
Created on Wed Sep 21 14:33:46 2022

@author: Jabed-Akhtar
******************************************************************************
"""
#Imports ======================================================================
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


#Importing Data ===============================================================
ticker = 'GOOG'
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2010-1-1')['Adj Close']
#Plot
data.plot(figsize=(15,6))


# ****************************** END OF FILE **********************************