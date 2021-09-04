# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:54:32 2020
Last edited September 4th 2021
@author: egemenokte

This code runs the MC Dropout model for pavement response prediction. It loads a test point and plots predictions on the test point. For different points, you can load another test point in pandas dataframe format. 
Required packages are listed in requirements.txt
"""

#https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf python code required to create dropout neural net

#%% Packages
import sys
import os
sys.path.append('../Manual Packages/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import sklearn
import seaborn as sns
from Data_Importer import importpickle
from Data_Importer import create_dropout_predict_function
from Data_Importer import plotsensitivitymultDROP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%%
TestPoint=importpickle('TestPoint') #Load the test point
print(TestPoint.head()) #display properties
Model=importpickle('MCDROPMODEL') #load the model
M = create_dropout_predict_function(Model['Mult'],0.1) #create dropout instance of the model with 0.1 dropout rate

plt.close('all')
sns.set(font_scale=1.5,rc={'figure.figsize':(18,9)}) #seaborn for better visuals (not required)

col=2 #2nd column in TestData is load to we will do a load sensitivity
start=20 #start sensitivity from 20 (kN)
stop=120 #start sensitivity from 120 (kN)
step=20 #step with 20 kN
plotsensitivitymultDROP(Model,TP=TestPoint,col=2,start=start,stop=stop,step=step,M=M,title='Error bounds with 2$\sigma$',xtitle = 'Load (kN)',ytitle='Response ($\mu$$\epsilon$)')
    

