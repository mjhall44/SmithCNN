import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate
import keras as keras
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import h5py
from datetime import datetime
import pandas as pd



def gammasweep(netname,modeldir, testdir):
	
    # evaluate mat files
    testingFiles = []
    data_dir = testdir
    for file in os.listdir(data_dir):  # pull all training files from directory
        if file.endswith(".mat"):
            testingFiles.append(file)
    nsets = len(testingFiles)  # number of training files
    
    model_path = pjoin(modeldir, netname)
    model = keras.models.load_model(model_path)
    print('Testing on ' + str(nsets) + ' files.')
    gammas = np.zeros([nsets, 1])
    for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()])

        
        ntimepts = np.size(waveforms,axis =1)-1
        x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
        y_test = waveforms[:, 0]  # waveform labels
        x_test = np.expand_dims(x_test, axis=2)
        predicts_CNN = model.predict(x_test,verbose =0)
        
        best_g_CNN = 0
        g_of_CNN = 0.2 #baseline value
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        noise_are = ranger[y_test == 0]
        
        for gamma in np.arange(0.01, 0.99, 0.01):
            
            predlabel_CNN = predicts_CNN[:,0] > gamma
            CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
            CNN_spike_acc = sum(y_test[spikes_are] == predlabel_CNN[spikes_are])/len(spikes_are)
            CNN_noise_acc = sum(y_test[noise_are] == predlabel_CNN[noise_are])/len(noise_are)
            if best_g_CNN  < (CNN_acc):
                best_g_CNN  = (CNN_acc)
                g_of_CNN  = gamma
            print("Gamma: %f\tCNN %f\tspikes:\tCNN %f\tnoise: %f" %(gamma, CNN_acc*100,  CNN_spike_acc*100, CNN_noise_acc*100))
            #print("%f\t%f\t%f\t%f" %(gamma, CNN_acc*100, CNN_spike_acc*100, CNN_noise_acc*100)) #easier format for copying to csv
        print("Best gamma -\tCNN: %f" %(g_of_CNN))
        predlabel_CNN = predicts_CNN[:,0] > g_of_CNN
        CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
        print("Best ACC:\tCNN %f" %(CNN_acc*100))
        gammas[j,0] = g_of_CNN
    bestg = np.mean(gammas)
    print("Best average gamma -\tCNN: %f" %(bestg))
    
    
    return bestg

def getPredicts(netname,modeldir, testdir, savedir=""):
    save = input("Save predicts as a .csv in this directory? (y/n): ")
    model_path = pjoin(modeldir, netname)
    model = keras.models.load_model(model_path)
    #evaluate mat files
    testingFiles = []
    data_dir = testdir
    for file in os.listdir(data_dir):  # pull all training files from directory
        if file.endswith(".mat"):
            testingFiles.append(file)
    nsets = len(testingFiles)  # number of training files
    
    model_path = pjoin(modeldir, netname)
    model = keras.models.load_model(model_path)
    print('Predicting p_spike for ' + str(nsets) + ' files.')
    data1 = pd.DataFrame()
    for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()])
        
        ntimepts = np.size(waveforms,axis =1)-1
        x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
        y_test = waveforms[:, 0]  # waveform labels
        x_test = np.expand_dims(x_test, axis=2)
        predicts_CNN = model.predict(x_test,verbose =0)
        data = pd.DataFrame({testingFiles[j]:np.squeeze(predicts_CNN)})
        data1 = pd.concat([data1,data], ignore_index=True, axis=1)
    data1.columns = testingFiles
    if save == 'y':
        date_time = datetime.now().strftime("%Y_%m_%d_%H-%M-%S_")
        savename = savedir+"testPredicts_" + date_time +netname+ ".csv"
        data1.to_csv(savename)
    return data1

def getLabels(netname,modeldir, testdir, gamma, savedir=""): 
    model_path = pjoin(modeldir, netname)
    model = keras.models.load_model(model_path)
    #evaluate mat files
    testingFiles = []
    data_dir = testdir
    for file in os.listdir(data_dir):  # pull all training files from directory
        if file.endswith(".mat"):
            testingFiles.append(file)
    nsets = len(testingFiles)  # number of training files
    
    model_path = pjoin(modeldir, netname)
    model = keras.models.load_model(model_path)
    print('Predicting Labels for ' + str(nsets) + ' files.')
    matsave = input("Save labels to .mat files as (filename_labeled.mat)? (y/n): ")
    save = input("Save labels as a .csv in this directory? (y/n): ")
    data1 = pd.DataFrame() #save predictions
    for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()])
        
        ntimepts = np.size(waveforms,axis =1)-1
        x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
        y_test = waveforms[:, 0]  # waveform labels
        x_test = np.expand_dims(x_test, axis=2)
        predicts_CNN = model.predict(x_test,verbose =0)
        predlabel_CNN = (predicts_CNN[:,0] > gamma)*1
        
        if matsave == 'y':
            waveData = np.concatenate((np.expand_dims(predlabel_CNN,axis =1), np.squeeze(x_test)), axis =1)
            tempname = testingFiles[j].replace('.mat','')
            savename = savedir+tempname+'_labeled'+ ".mat"
            sio.savemat(savename, {'waveData':waveData})
        data = pd.DataFrame({testingFiles[j]:(predlabel_CNN)})
        data1 = pd.concat([data1,data], ignore_index=True, axis=1)
    data1.columns = testingFiles
    
    if save == 'y':
        date_time = datetime.now().strftime("%Y_%m_%d_%H-%M-%S_")
        savename = savedir+"testLabels_" + date_time +netname+ ".csv"
        data1.to_csv(savename)
    return data1