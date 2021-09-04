# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:48:01 2020

@author: egeme
used to import xlsx into dict format
"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
def importxlsx(filepath) :
    ##If picklefilename is 0, it does not pickle
    print('Getting Files Ready') 
    excel_file_str = filepath #structure info
    # this will read the first sheet into df
    
    print('Reading Files. Please be patient') 
    xls = pd.ExcelFile(excel_file_str)

    
    # Get The names
    Names = xls.sheet_names
    # to read all sheets to a map so that it can be accessed later
    Data= {}
    print('Importing Data') 
    totallen=len(Names)
    c=0
    for sheet_name in Names: #for structure
        Data[sheet_name] = xls.parse(sheet_name)
        c=c+1
        print('Data Import ' ,str(round(c/totallen*100)),'%' )
        
    return Data

def importpickle(filepath) : #imports in picklefiles

    filename = '../PickleData/'+filepath
    infile = open(filename,'rb')
    Data = pickle.load(infile)
    infile.close()
        
    return Data

def importpicklepath(filepath) : #imports in picklefiles

    filename = filepath
    infile = open(filename,'rb')
    Data = pickle.load(infile)
    infile.close()
        
    return Data

def exportpickle(Data,filepath) : #imports in picklefiles

    filename = '../PickleData/'+filepath
    outfile = open(filename,'wb')
    pickle.dump(Data,outfile)
    outfile.close() 
    print('Data Exported') 
        
    return 


def create_dropout_predict_function(model, dropout):
    from keras.models import Model, Sequential
    from keras import backend as K
    """
    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"]=="Dropout":
            layer["config"]["rate"] = dropout
        # Recurrent layers with dropout
        elif "dropout" in layer["config"].keys():
            layer["config"]["dropout"] = dropout

    # Create a new model with specified dropout
    if type(model)==Sequential:
        # Sequential
        model_dropout = Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    # Create a function to predict with the dropout on
    predict_with_dropout = K.function(model_dropout.inputs+[K.learning_phase()], model_dropout.outputs)
    
    return predict_with_dropout

def plotsensitivitymultDROP(Model,TP,col,start,stop,step,M,title=' ',xtitle = '',ytitle='MicroStrains'): 
    '''
    plots sensitivity given a column, stop, end and step values linearly

    Returns
    -------
    None.

    '''
    NAMES=['$\epsilon_{11}^{surf}$','$\epsilon_{33}^{surf}$','$\epsilon_{11}^{botac}$','$\epsilon_{33}^{botac}$','$\epsilon_{22}^{ac}$','$\epsilon_{22}^{base}$','$\epsilon_{22}^{sg}$']
    predprog=[]
    TT=TP.iloc[0,:].values.reshape(1,-1)
    TT = np.repeat(TT,step,axis=0)
    #L=np.linspace(TT[0,1],100,20)
    L=np.linspace(start,stop,step)
    TT[:,col]=L
    Xt2 = Model['Transformer'].transform(TT)
    # Xt2 = poly.transform(Xt2)
    num_iter=100
    for i in range(1):
        p=Model['TransformerY'].inverse_transform(Model['Mult'].predict(Xt2))
        predprog.append(p)
        predictions2 = np.zeros((len(L),7, num_iter))
        for i in range(num_iter):
            predictions2[:,:,i] = Model['TransformerY'].inverse_transform(M([Xt2,1]+[1])[0])
        stds=np.std(predictions2,axis=2)
    fig, axs = plt.subplots(2,4)
    fig.suptitle(title)
    ind=0
    for i in range(2):
        for j in range(4):
            # axs[i,j].scatter(L,predprog[0][:,ind])
            # axs[i,j].scatter(Model['Test_X'][n,col],Model['Test_Y'][n,ind],color='b')
            # axs[i,j].errorbar(L,np.mean(predictions2,axis=2)[:,ind],yerr=2*stds[:,ind],fmt='o',errorevery=1)
            y=np.mean(predictions2,axis=2)[:,ind]
            error=2*stds[:,ind]
            axs[i,j].plot(L,y, marker='o',linewidth=2,markersize=5)
            axs[i,j].fill_between(L, y-error, y+error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            axs[i,j].set_title(NAMES[ind])
            ind=ind+1
            if ind>=7:
                break
    fig.text(0.5, 0.04, xtitle, ha='center')
    fig.text(0.04, 0.5, ytitle, va='center', rotation='vertical')
    return
    # fig.savefig('prediction.png')
    


