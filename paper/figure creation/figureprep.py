#Figure creation for CNN and NAS comparison paper

# Date Created: 09-11-2021
# Created by Matthew Hall
# Last Edited:  04-03-2022

import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate

import keras as keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Dropout

import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import h5py

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.pyplot
from scipy.fft import fft, fftfreq

from datetime import datetime
import random

# Networks to be tested

model_dir = 'Z:\\SmithLab\\prjNAS\\NASNet'


limiter =150
gamma = 0.15
gammas = np.arange(1,10)/10
gammas =0.15
numLines =800
ntimepts = 52

netnames =[]
numcnn =1

netnames.append('SmithCNNnet') #best trained network
netnames.append('NAS_full') #original nas network

# evaluate mat files as one large chunk
#data returned was used to determine the realtime test across a random sample of all cortical areas

data_dir = 'Z:/SmithLab/prjNAS/NASNet/test_all/' #all areas in one file
testingFile = []
for file in os.listdir(data_dir):  # pull all training files from directory
    if file.endswith(".mat"):
        testingFile.append(file)
nset = len(testingFile)  # singular testing file

#Time testing on full file (random choice over all areas)

#print('Testing on ' + str(nsets) + ' files.')
for j in range(nset):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFile[j],mode='r')  # load testing file(s)
        testacc = []
        fozle = testingFile[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
        #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
        for netname in netnames:
            match = netnames.index(netname)
            model_path = pjoin(model_dir, netname)
            model = keras.models.load_model(model_path)

            if match < numcnn:

                
                
                x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                y_test = waveforms[:, 0]  # waveform labels
                x_test = np.expand_dims(x_test, axis=2)
                #predicts_CNN = model.predict(x_test,verbose =0)
                #print(netname +area+" has acc of: "+str(predicts) +" with " +str(model.count_params()) + " params.")
                #print(predicts_CNN.shape)
                
                
                #timetest
                time_diffs = []
                for i in np.arange(1,100):
                    x_test_400 = x_test[np.random.choice(len(waveforms),400),:]
                    t1 = datetime.now().now()
                    predicts_scrub = model.predict(x_test_400,verbose =0)
                    t2 = datetime.now().now()
                    time_diff = (t2-t1)/400
                    time_diffs.append(time_diff.total_seconds())
                
                
                print(f'Time Diff CNN: {np.mean(time_diffs)} var: {np.var(time_diffs)}')
            else:
                #print('we made it')
                x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                #x_test = np.expand_dims(x_test, axis=2)
                y_test = waveforms[:, 0]  # waveform labels
                
                #predicts_NAS = model.predict(x_test,verbose =0)
                #print(netname +area+" has acc of: "+str(predicts) +" with " +str(model.count_params()) + " params.")
                #print(predicts_NAS.shape)
                
                #timetest
                time_diffs = []
                for i in np.arange(1,100):
                    x_test_400 = x_test[np.random.choice(len(waveforms),400),:]
                    t1 = datetime.now().now()
                    predicts_scrub = model.predict(x_test_400,verbose =0)
                    t2 = datetime.now().now()
                    time_diff = (t2-t1)/400
                    time_diffs.append(time_diff.total_seconds())
                
                
                print(f'Time Diff NAS: {np.mean(time_diffs)} var: {np.var(time_diffs)}')
                
                # NAS 2.144 times faster than CNN at 113.52% of the variance
                # CNN classifies 6907 waveforms/sec
                # NAS classifies 14805 waveforms/sec


#figure 3 -  gamma sweeps and accuracy across each area
# data was placed into an excel spreadsheet
data_dir = 'Z:/SmithLab/prjNAS/NASNet/tests_byarea/' #all areas in separate files
testingFiles = []
for file in os.listdir(data_dir):  # pull all training files from directory
    if file.endswith(".mat"):
        testingFiles.append(file)
nsets = len(testingFiles)  # number of training files

for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        testacc = []
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
        #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
        for netname in netnames:
            match = netnames.index(netname)
            model_path = pjoin(model_dir, netname)
            model = keras.models.load_model(model_path)

            if match < numcnn:

                
                
                x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                y_test = waveforms[:, 0]  # waveform labels
                x_test = np.expand_dims(x_test, axis=2)
                predicts_CNN = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_CNN) +" with " +str(model.count_params()) + " params.")
                #print(predicts_CNN.shape)
                
                
            else:
                #print('we made it')
                x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                #x_test = np.expand_dims(x_test, axis=2)
                y_test = waveforms[:, 0]  # waveform labels
                
                predicts_NAS = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_NAS) +" with " +str(model.count_params()) + " params.")
                #print(predicts_NAS.shape)
                
        best_g_NAS = 0
        best_g_CNN = 0
        g_of_CNN = 0.2
        g_of_NAS = 0.2
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        noise_are = ranger[y_test == 0]
        
        for gamma in np.arange(0.01, 0.99, 0.01):
            
            predlabel_CNN = predicts_CNN[:,0] > gamma
            predlabel_NAS = predicts_NAS[:,0] > (gamma)
            CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
            NAS_acc = sum(y_test == predlabel_NAS)/len(y_test)
            ranger
            
            #CNN_spike_acc = sum(y_test[spikes_are] == predlabel_CNN[spikes_are])/len(spikes_are)
            #NAS_spike_acc = sum(y_test[spikes_are] == predlabel_NAS[spikes_are])/len(spikes_are)
            CNN_noise_acc = sum(y_test[noise_are] == predlabel_CNN[noise_are])/len(noise_are)
            NAS_noise_acc = sum(y_test[noise_are] == predlabel_NAS[noise_are])/len(noise_are)
            if best_g_CNN  < (CNN_acc):
                best_g_CNN  = (CNN_acc)
                g_of_CNN  = gamma
            if best_g_NAS  < (NAS_acc):
                best_g_NAS  = (NAS_acc)
                g_of_NAS  = gamma
            #print("Gamma: %f\tCNN %f\tNAS %f spikes:\tCNN %f\tNAS %f" %(gamma, CNN_acc*100, NAS_acc*100, CNN_spike_acc*100, NAS_spike_acc*100))
            #print("%f\t%f\t%f\t%f\t%f" %(gamma, CNN_acc*100, NAS_acc*100, CNN_spike_acc*100, NAS_spike_acc*100))
            print("%f\t%f\t%f" %(gamma, CNN_noise_acc*100, NAS_noise_acc*100))
        

        print("Best gamma -\tCNN: %f \tNAS: %f" %(g_of_CNN, g_of_NAS))
        #gamma = g_of
        
        predlabel_CNN = predicts_CNN[:,0] > g_of_CNN
        predlabel_NAS = predicts_NAS[:,0] > (g_of_NAS)
        CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
        NAS_acc = sum(y_test == predlabel_NAS)/len(y_test)
        print("Best ACC:\tCNN %f\tNAS %f" %(CNN_acc*100, NAS_acc*100))
        
        best_g_NAS = 0
        best_g_CNN = 0
        g_of_CNN = 0.48
        g_of_NAS = 0.48
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        print("Best gamma -\tCNN: %f \tNAS: %f" %(g_of_CNN, g_of_NAS))
        
        predlabel_CNN = predicts_CNN[:,0] > g_of_CNN
        predlabel_NAS = predicts_NAS[:,0] > (g_of_NAS)
        CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
        NAS_acc = sum(y_test == predlabel_NAS)/len(y_test)
        print("Best ACC:\tCNN %f\tNAS %f" %(CNN_acc*100, NAS_acc*100))

#figure 3 on excel CNNnas.xlsx
#range and variance 
# 10 fold cross validation on test data
data_dir_base = 'Z:/SmithLab/prjNAS/NASNet/tests_byarea_kfold/' #all areas in separate files

area_names = ["PFC","M1","V4", "FEF"]
#area_names = ["V4"]
area_var = np.zeros([4,4])
for area in area_names:
    data_dir = data_dir_base + area+ '/'
    testingFiles = []
    for file in os.listdir(data_dir):  # pull all training files from directory
        if file.endswith(".mat"):
            testingFiles.append(file)
    nsets = len(testingFiles)  # number of training files
    print(area)
    kfold = np.zeros([nsets, 4]) #contains accuracy at 0.2 for NAS/CNN, and 0.48 for NAS/CNN
    for j in range(nsets):  # loop over the different testing files
            f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
            testacc = []
            fozle = testingFiles[j]
            file_edit = fozle.rfind('_')
            datastr   = fozle[file_edit+1: len(fozle)-4]
            print(fozle)
            waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
            #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
            for netname in netnames:
                match = netnames.index(netname)
                model_path = pjoin(model_dir, netname)
                model = keras.models.load_model(model_path)

                if match < numcnn:
                    
                    x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                    y_test = waveforms[:, 0]  # waveform labels
                    x_test = np.expand_dims(x_test, axis=2)
                    predicts_CNN = model.predict(x_test,verbose =0)


                else:
                    #print('we made it')
                    x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                    #x_test = np.expand_dims(x_test, axis=2)
                    y_test = waveforms[:, 0]  # waveform labels
                    predicts_NAS = model.predict(x_test,verbose =0)
                    
            predlabel_CNNpt2 = predicts_CNN[:,0] > 0.2
            predlabel_CNNpt48 = predicts_CNN[:,0] > 0.48
            predlabel_NASpt2 = predicts_NAS[:,0] > 0.2
            predlabel_NASpt48 = predicts_NAS[:,0] > 0.48
            kfold[j, 0] = sum(y_test == predlabel_NASpt2)/len(y_test)
            kfold[j, 1] = sum(y_test == predlabel_CNNpt2)/len(y_test)
            kfold[j, 2] = sum(y_test == predlabel_NASpt48)/len(y_test)
            kfold[j, 3] = sum(y_test == predlabel_CNNpt48)/len(y_test)
            
    area_var[area_names.index(area)] =  [np.var(kfold[:, 0]), np.var(kfold[:, 1]), np.var(kfold[:, 2]),np.var(kfold[:, 3])]

    print("gamma=0.2 \t\tNAS\t\tCNN\t\tgamma=0.48 NAS\t\tCNN")
    print("average acc: ", np.mean(kfold[:, 0]), np.mean(kfold[:, 1]), np.mean(kfold[:, 2]),np.mean(kfold[:, 3]))
    print("var:\t    ", np.var(kfold[:, 0]), np.var(kfold[:, 1]), np.var(kfold[:, 2]),np.var(kfold[:, 3])) 

print(area_names)
print(area_var)
print("gamma=0.2 \t\tNAS\t\tCNN\t\tgamma=0.48 NAS\t\tCNN")
print("average var: ", np.mean(area_var[:, 0]), np.mean(area_var[:, 1]), np.mean(area_var[:, 2]),np.mean(area_var[:, 3]))

# buffer load all predictions here
#figure 3 -  gamma sweeps and accuracy across each area
data_dir = 'Z:/SmithLab/prjNAS/NASNet/test_all/' #all areas in one file
testingFiles = []
for file in os.listdir(data_dir):  # pull all training files from directory
    if file.endswith(".mat"):
        testingFiles.append(file)
nsets = len(testingFiles)  # singular testing file

for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        testacc = []
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
        #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
        for netname in netnames:
            match = netnames.index(netname)
            model_path = pjoin(model_dir, netname)
            model = keras.models.load_model(model_path)

            if match < numcnn:

                
                
                x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                y_test = waveforms[:, 0]  # waveform labels
                x_test = np.expand_dims(x_test, axis=2)
                predicts_CNNall = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_CNNall) +" with " +str(model.count_params()) + " params.")
                #print(predicts_CNN.shape)
                
                
            else:
                #print('we made it')
                x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                #x_test = np.expand_dims(x_test, axis=2)
                y_test = waveforms[:, 0]  # waveform labels
                
                predicts_NASall = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_NASall) +" with " +str(model.count_params()) + " params.")
                #print(predicts_NAS.shape)
                
        best_g_NAS = 0
        best_g_CNN = 0
        g_of_CNN = 0.2
        g_of_NAS = 0.2
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        noise_are = ranger[y_test == 0]
        
        for gamma in np.arange(0.01, 0.99, 0.01):
            
            predlabel_CNN = predicts_CNNall[:,0] > gamma
            predlabel_NAS = predicts_NASall[:,0] > (gamma)
            CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
            NAS_acc = sum(y_test == predlabel_NAS)/len(y_test)
            ranger
            
            #CNN_spike_acc = sum(y_test[spikes_are] == predlabel_CNN[spikes_are])/len(spikes_are)
            #NAS_spike_acc = sum(y_test[spikes_are] == predlabel_NAS[spikes_are])/len(spikes_are)
            CNN_noise_acc = sum(y_test[noise_are] == predlabel_CNN[noise_are])/len(noise_are)
            NAS_noise_acc = sum(y_test[noise_are] == predlabel_NAS[noise_are])/len(noise_are)
            if best_g_CNN  < (CNN_acc):
                best_g_CNN  = (CNN_acc)
                g_of_CNN  = gamma
            if best_g_NAS  < (NAS_acc):
                best_g_NAS  = (NAS_acc)
                g_of_NAS  = gamma
            #print("Gamma: %f\tCNN %f\tNAS %f spikes:\tCNN %f\tNAS %f" %(gamma, CNN_acc*100, NAS_acc*100, CNN_spike_acc*100, NAS_spike_acc*100))
            #print("%f\t%f\t%f\t%f\t%f" %(gamma, CNN_acc*100, NAS_acc*100, CNN_spike_acc*100, NAS_spike_acc*100))
            print("%f\t%f\t%f" %(gamma, CNN_noise_acc*100, NAS_noise_acc*100))
        

        print("Best gamma -\tCNN: %f \tNAS: %f" %(g_of_CNN, g_of_NAS))
        #gamma = g_of
        
        predlabel_CNN = predicts_CNNall[:,0] > g_of_CNN
        predlabel_NAS = predicts_NASall[:,0] > (g_of_NAS)
        CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
        NAS_acc = sum(y_test == predlabel_NAS)/len(y_test)
        print("Best ACC:\tCNN %f\tNAS %f" %(CNN_acc*100, NAS_acc*100))
        
        best_g_NAS = 0
        best_g_CNN = 0
        g_of_CNN = 0.48
        g_of_NAS = 0.48
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        print("Best gamma -\tCNN: %f \tNAS: %f" %(g_of_CNN, g_of_NAS))
        
        predlabel_CNN = predicts_CNNall[:,0] > g_of_CNN
        predlabel_NAS = predicts_NASall[:,0] > (g_of_NAS)
        CNN_acc = sum(y_test == predlabel_CNN)/len(y_test)
        NAS_acc = sum(y_test == predlabel_NAS)/len(y_test)
        print("Best ACC:\tCNN %f\tNAS %f" %(CNN_acc*100, NAS_acc*100))
        
        
        max_idx = []
        X_maxes = np.amax(x_test,1)
        pos = 0
        for mx in X_maxes:
            max_val = np.where(x_test[pos,:] == mx)
            max_val_val = max_val[0]
            max_idx.append(max_val_val[0])
            pos = pos + 1
        min_idx = []
        X_mins = np.amin(x_test,1)
        pos = 0
        for mins in X_mins:
            min_val = np.where(x_test[pos,:] == mins)
            min_val_val = min_val[0]
            min_idx.append(min_val_val[0])
            pos = pos + 1
        y_test_idx= []
        y_test_idx_noise= []
        max_idx_spike = []

        p_CNN_spike = []
        p_NAS_spike = []
        p_CNN_noise = []
        p_NAS_noise = []
        max_idx_noise = []
        for p in np.arange(0,len(y_test)-1):
            if y_test[p] == 1:
                y_test_idx = p
                max_idx_spike.append(max_idx[y_test_idx])
                p_CNN_spike.append(predicts_CNNall[y_test_idx,0])
                p_NAS_spike.append(predicts_NASall[y_test_idx,0])
            else:
                y_test_idx_noise = p
                max_idx_noise.append(max_idx[y_test_idx_noise])
                p_CNN_noise.append(predicts_CNNall[y_test_idx_noise,0])
                p_NAS_noise.append(predicts_NASall[y_test_idx_noise,0])
# supplemental figure
# Distribution of spike probability for each Network
p_CNN_spike = []
p_NAS_spike = []
p_CNN_noise = []
p_NAS_noise = []

plt.figure()
kwargs = dict(alpha=0.5, bins=50)
plt.hist(predicts_CNNall[:,0], **kwargs, label = "CNN",color='b')
plt.hist(predicts_NASall[:,0], **kwargs, label = "NAS",color='y')
plt.legend()
plt.xlabel("Spike Probability")
plt.ylabel("Bin Count")
plt.title("Distribution of Spike Probability for Each Network")
plt.xlim([0, 1])
plt.ticklabel_format(useOffset=False, style='plain')
plt.show()

#figure 4 - comparison to hand label distribution
#over a single chanel of v4
data_dir = 'Z:/SmithLab/prjNAS/NASNet/tests_m1/v4_ch/' #all areas in separate files
testingFiles = []
for file in os.listdir(data_dir):  # pull all training files from directory
    if file.endswith(".mat"):
        testingFiles.append(file)
nsets = len(testingFiles)  # number of training files

for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        testacc = []
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
        #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
        for netname in netnames:
            match = netnames.index(netname)
            model_path = pjoin(model_dir, netname)
            model = keras.models.load_model(model_path)

            if match < numcnn:

                
                
                x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                y_test = waveforms[:, 0]  # waveform labels
                x_test = np.expand_dims(x_test, axis=2)
                predicts_CNN = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_CNN) +" with " +str(model.count_params()) + " params.")
                #print(predicts_CNN.shape)
                
                
            else:
                #print('we made it')
                x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                #x_test = np.expand_dims(x_test, axis=2)
                y_test = waveforms[:, 0]  # waveform labels
                
                predicts_NAS = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_NAS) +" with " +str(model.count_params()) + " params.")
                #print(predicts_NAS.shape)
                
        best_g_NAS = 0
        best_g_CNN = 0
        g_of_CNN = 0.2
        g_of_NAS = 0.2
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        noise_are = ranger[y_test == 0]
        
        lims = 200
        #ch = fozle[fozle.find('_SNR')-2:fozle.find('_SNR')]
        ch = fozle[fozle.find('ch')+2:fozle.find('ch')+3]
        lab = 'V4'
        rngvals = np.random.choice(len(predicts_NAS),3000)
        t = np.arange(0,52)
        plt.figure(figsize=(8, 6),dpi=100)
        for r in rngvals:
            if y_test[r] == 1:
                plt.plot(t,x_test[r,:],c='darkgreen', alpha=0.8)
            else:
                plt.plot(t,x_test[r,:],c='darkred', alpha=0.6)
        plt.title('Hand Labeled Data for ' +lab)
        plt.ylabel(u'\u03bcV')
        plt.xlim([0, 51])
        plt.ylim([-lims ,lims])
        custom_lines = [mpl.lines.Line2D([0], [0], color='g', lw=4),
                mpl.lines.Line2D([0], [0], color='brown', lw=4)]

        plt.legend(custom_lines, ['Spike ' + str(round(sum(y_test)*100/len(y_test)))+'%', 'Noise '+ str(round(100-sum(y_test)*100/len(y_test)))+'%'])
        plt.show()


        n_lines = 10
        c = np.arange(1, n_lines + 1)

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn')
        cmap.set_array([])

        fig, ax = plt.subplots(figsize=(8, 6),dpi=100)
        for r in rngvals:
            ax.plot(t, x_test[r,:], c=cmap.to_rgba(predicts_NAS[r,0]))
        #fig.colorbar(cmap)

        plt.title('NAS predictions for ' +lab +' When Removed from Training')
        plt.ylabel(u'\u03bcV')
        plt.xlim([0, 51])
        plt.ylim([-lims ,lims])
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        for r in rngvals:
            ax.plot(t, x_test[r,:], c=cmap.to_rgba(predicts_CNN[r,0]))
        cbar = fig.colorbar(cmap)
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        cbar.ax.set_ylabel('$p_{spike}$', rotation=270)
        plt.title('CNN predictions for ' +lab+' When Removed from Training')
        plt.ylabel(u'\u03bcV')
        plt.xlim([0, 51])
        plt.ylim([-lims,lims])
        plt.show()

#figure 4 m1 plots
# uses networks that were trained on lacking m1 data

data_dir = 'Z:/SmithLab/prjNAS/NASNet/small_m1/' #all areas in separate files
testingFiles = []
for file in os.listdir(data_dir):  # pull all training files from directory
    if file.endswith(".mat"):
        testingFiles.append(file)
nsets = len(testingFiles)  # number of training files

netnames_alt = ['CNN4L2d_opt_noM1','NAS_noM1']
for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        testacc = []
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
        #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
        for netname in netnames_alt:
            match = netnames_alt.index(netname)
            model_path = pjoin(model_dir, netname)
            model = keras.models.load_model(model_path)

            if match < numcnn:

                
                
                x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                y_test = waveforms[:, 0]  # waveform labels
                x_test = np.expand_dims(x_test, axis=2)
                predicts_CNN = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_CNN) +" with " +str(model.count_params()) + " params.")
                #print(predicts_CNN.shape)
                
                
            else:
                #print('we made it')
                x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                #x_test = np.expand_dims(x_test, axis=2)
                y_test = waveforms[:, 0]  # waveform labels
                
                predicts_NAS = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_NAS) +" with " +str(model.count_params()) + " params.")
                #print(predicts_NAS.shape)
                
        best_g_NAS = 0
        best_g_CNN = 0
        g_of_CNN = 0.2
        g_of_NAS = 0.2
        ranger = np.arange(0,len(y_test),1)
        spikes_are = ranger[y_test == 1]
        noise_are = ranger[y_test == 0]
        
        lims = 200
        #ch = fozle[fozle.find('_SNR')-2:fozle.find('_SNR')]
        ch = fozle[fozle.find('ch')+2:fozle.find('ch')+3]
        lab = 'M1'
        rngvals = np.random.choice(len(predicts_NAS),3000)
        t = np.arange(0,52)
        plt.figure(figsize=(8, 6),dpi=100)
        for r in rngvals:
            if y_test[r] == 1:
                plt.plot(t,x_test[r,:],c='darkgreen', alpha=0.8)
            else:
                plt.plot(t,x_test[r,:],c='darkred', alpha=0.6)
        plt.title('Hand Labeled Data for ' +lab)
        plt.ylabel(u'\u03bcV')
        plt.xlim([0, 51])
        plt.ylim([-lims ,lims])
        custom_lines = [mpl.lines.Line2D([0], [0], color='g', lw=4),
                mpl.lines.Line2D([0], [0], color='brown', lw=4)]

        plt.legend(custom_lines, ['Spike ' + str(round(sum(y_test)*100/len(y_test)))+'%', 'Noise '+ str(round(100-sum(y_test)*100/len(y_test)))+'%'])
        plt.show()


        n_lines = 10
        c = np.arange(1, n_lines + 1)

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn')
        cmap.set_array([])

        fig, ax = plt.subplots(figsize=(8, 6),dpi=100)
        for r in rngvals:
            ax.plot(t, x_test[r,:], c=cmap.to_rgba(predicts_NAS[r,0]))
        #fig.colorbar(cmap)

        plt.title('NAS predictions for ' +lab +' When Removed from Training')
        plt.ylabel(u'\u03bcV')
        plt.xlim([0, 51])
        plt.ylim([-lims ,lims])
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        for r in rngvals:
            ax.plot(t, x_test[r,:], c=cmap.to_rgba(predicts_CNN[r,0]))
        cbar = fig.colorbar(cmap)
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        cbar.ax.set_ylabel('$p_{spike}$', rotation=270)
        plt.title('CNN predictions for ' +lab+' When Removed from Training')
        plt.ylabel(u'\u03bcV')
        plt.xlim([0, 51])
        plt.ylim([-lims,lims])
        plt.show()

data_dir = 'Z:/SmithLab/prjNAS/NASNet/test_all/' #all areas in one file
testingFiles = []
for file in os.listdir(data_dir):  # pull all training files from directory
    if file.endswith(".mat"):
        testingFiles.append(file)
nsets = len(testingFiles)  # singular testing file

for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(data_dir + '/' + testingFiles[j],mode='r')  # load testing file(s)
        testacc = []
        fozle = testingFiles[j]
        file_edit = fozle.rfind('_')
        datastr   = fozle[file_edit+1: len(fozle)-4]
        print(fozle)
        waveforms = np.transpose(f_waves['waveData'][()]) #load all waveforms
        #waveforms = waveforms[np.random.choice(len(waveforms),5000000), :]
        for netname in netnames:
            match = netnames.index(netname)
            model_path = pjoin(model_dir, netname)
            model = keras.models.load_model(model_path)

            if match < numcnn:

                
                
                x_test = waveforms[:, 1:(52+1)]  # waveform voltage values
                y_test = waveforms[:, 0]  # waveform labels
                x_test = np.expand_dims(x_test, axis=2)
                #predicts_CNNall = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_CNNall) +" with " +str(model.count_params()) + " params.")
                #print(predicts_CNN.shape)
                
                
            else:
                #print('we made it')
                x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
                #x_test = np.expand_dims(x_test, axis=2)
                y_test = waveforms[:, 0]  # waveform labels
                
                predicts_NASall = model.predict(x_test,verbose =0)
                print(netname +" has acc of: "+str(predicts_NASall) +" with " +str(model.count_params()) + " params.")
                #print(predicts_NAS.shape)
#figure 5 here, classification of all areas
#histogram distribution of waveform/spike max and accuracy on that specific bin index
gamma=0.48

predlabel_CNN = predicts_CNNall[:,0] > gamma
predlabel_NAS = predicts_NASall[:,0] > (gamma)
# Get the histogramp
Y,X = np.histogram(max_idx, 52,range=((0,51)))
bins =52
t_waves = np.zeros(bins)
cor_waves = np.zeros((bins,2))
err_waves = np.zeros(bins)
for maxW in np.arange(0,52):
    idx =0
    for maxval in max_idx:
        t_waves[maxval] = t_waves[maxval] + 1
        if predlabel_CNN[idx] == y_test[idx]:
            cor_waves[maxval,1] = cor_waves[maxval,1] + 1
        if predlabel_NAS[idx] == y_test[idx]:
            cor_waves[maxval,0] = cor_waves[maxval,0] + 1    
        idx = idx + 1

Ys,Xs = np.histogram(max_idx_spike, 52,range=((0,51)))
bins =52
t_spike = np.zeros(bins)
cor_spike = np.zeros((bins,2))
for maxW in np.arange(0,52):
    idx =0
    for maxval in max_idx:
        if y_test[idx] == 1:
            t_spike[maxval] = t_spike[maxval] + 1
            if predlabel_CNN[idx] == y_test[idx]:
                cor_spike[maxval,1] = cor_spike[maxval,1] + 1
            if predlabel_NAS[idx] == y_test[idx]:
                cor_spike[maxval,0] = cor_spike[maxval,0] + 1    
        idx = idx + 1

n, bins, patches = plt.hist(max_idx, 25, color='green')
bin_centers = 0.5 * (bins[:-1] + bins[1:])
plt.show()

#binned accuracy for each network type over relative waveform max
p_NAS = cor_waves[:,0]/t_waves
p_CNN = cor_waves[:,1]/t_waves
'''
'''
norm1 = p_NAS/np.linalg.norm(p_NAS)
norm2 = p_CNN/np.linalg.norm(p_CNN)
col = bin_centers - min(bin_centers)
col /= max(col)
c_map = plt.cm.get_cmap('binary', 15)
p_both = np.concatenate([p_NAS, p_CNN])
cm = c_map(p_both)



norm1 = p_NAS/np.linalg.norm(p_NAS)
norm2 = p_CNN/np.linalg.norm(p_CNN)
col = bin_centers - min(bin_centers)
col /= max(col)
c_map = plt.cm.get_cmap('binary', 15)
p_both = np.concatenate([p_NAS, p_CNN])
cm = c_map(p_both)
# histograms and delta accuracies
Y,X = np.histogram(max_idx, 52, range=(0,52))
x_span = X.max()-X.min()
bin_centers = 0.5 * (X[:-1] + X[1:])       
#ax = plt.bar(X[:-1],Y,color=cm[0:51,:],width=X[1]-X[0])
ax = plt.bar(X[:-1],Y,width=X[1]-X[0],edgecolor='black', linewidth=1.2)
plt.xlim([0-0.5,51+0.5])
#sm = plt.cm.ScalarMappable(cmap=c_map, norm=plt.Normalize(min(p_both),max(p_both)))
#sm.set_array([])

#cbar = plt.colorbar(sm)
plt.title("Distribution of Wave Max")
plt.xlabel('Wave max')
plt.ylabel('Bin Count')
plt.show()

plt.figure()
idxa =0
CNNbetterc =0
for p in p_CNN:
    if p-p_NAS[idxa] > 0:
        plt.scatter(idxa,p-p_NAS[idxa],c='b')
        CNNbetterc = CNNbetterc+1
    else:
        plt.scatter(idxa,p-p_NAS[idxa],c='y')
    idxa = idxa+1
#plt.plot(np.arange(0,52),p_CNN-p_NAS,'b')  
plt.plot(np.arange(0,52),np.zeros([52, 1]),'r--')
plt.title(u"Delta Accuracy for CNN-NAS at \u03B3 = 0.48")
plt.xlabel('Wave Max')
plt.ylabel(u'\u0394 Accuracy')
legend_elements = [mpl.lines.Line2D([0], [0], color='r', linestyle='--', label='Same Accuracy'),
    mpl.lines.Line2D([0], [0], color='w', marker='o', label='CNN better', markerfacecolor='b'),
           mpl.lines.Line2D([0], [0], marker='o', color='w', label='NAS better', markerfacecolor='y')]

plt.legend(handles=legend_elements, loc='upper right')
plt.xlim([0, 51])
plt.show()
print(CNNbetterc)

#binned accuracy for each network type over relative spike waveform max
p_NAS = cor_spike[:,0]/t_spike
p_CNN = cor_spike[:,1]/t_spike

norm1 = p_NAS/np.linalg.norm(p_NAS)
norm2 = p_CNN/np.linalg.norm(p_CNN)
col = bin_centers - min(bin_centers)
col /= max(col)
c_map = plt.cm.get_cmap('binary', 15)
p_both = np.concatenate([p_NAS, p_CNN])
cm = c_map(p_both)


Y,X = np.histogram(max_idx_spike, 52, range=(0,52))
x_span = X.max()-X.min()

ax = plt.bar(X[:-1],Y,width=X[1]-X[0],edgecolor='black', linewidth=1.2)
plt.xlim([0-0.5,51+0.5])
#sm = plt.cm.ScalarMappable(cmap=c_map, norm=plt.Normalize(min(p_both),max(p_both)))
#sm.set_array([])

plt.title("Spikes Only Wave Max distribution")
plt.xlabel('Wave max')
plt.ylabel('Bin Count')
plt.show()
'''

'''
plt.figure()
idxa =0
CNNbetterc =0
for p in p_CNN:
    if p-p_NAS[idxa] > 0:
        plt.scatter(idxa,p-p_NAS[idxa],c='b')
        CNNbetterc = CNNbetterc+1
    else:
        plt.scatter(idxa,p-p_NAS[idxa],c='y')
    idxa = idxa+1
#plt.plot(np.arange(0,52),p_CNN-p_NAS,'b')  
plt.plot(np.arange(0,52),np.zeros([52, 1]),'r--')


plt.title(u"Delta Accuracy for CNN-NAS at \u03B3 = 0.48 (Spikes Only)")
plt.xlabel('Wave Max')
plt.ylabel(u'\u0394 Accuracy')
#plt.legend(['Same Accuracy','CNN better','NAS better'])

legend_elements = [mpl.lines.Line2D([0], [0], color='r', linestyle='--', label='Same Accuracy'),
    mpl.lines.Line2D([0], [0], color='w', marker='o', label='CNN better', markerfacecolor='b'),
           mpl.lines.Line2D([0], [0], marker='o', color='w', label='NAS better', markerfacecolor='y')]

plt.legend(handles=legend_elements, loc='upper right')
plt.xlim([0, 51])
plt.show()
print(CNNbetterc)

