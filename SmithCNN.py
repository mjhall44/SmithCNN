import numpy as np
import tensorflow as tf
import random as rn
import h5py
import os
import keras as keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Dropout
from datetime import datetime


def trainnet(netname,ntimepts,traindir, testdir):
    #  trainnet trains a neural network with ntimepts # of units in the hidden layer and 1 unit in the output layer.
    #  Tested with Python3 version 3.7.4, pip3 version 19.2.3, virtualenv version 16.7.5, and tensorflow 2.0
    #  INPUTS-
    #   1. netname: network name (string)
    #   2. ntimepts: the # of time points in a single waveform (integer)
    #   3. traindir: the path location of the training files (string), ex. "/Users/NASNet/training dir"
    #   .............each file in the traindir should contain a N x (1 + ntimepts) matrix
    #   .............The rows are the N waveforms to train the model with.
    #   .............The rows are the N waveforms to train the model with.
    #   .............The first column of the matrix should be the BINARY waveform labels (0 for noise, 1 for spike)
    #   .............The remaining ntimepts columns are the waveform voltage values.
    #   .............The array must be stored under the variable/group name 'waveData'
    #  OUTPUT: this function saves the following files
    #   1. the trained network (a Keras model file)
    #   2. the network weights/biases (4 text files- 1 weight file + 1 bias file for both the hidden + output layers)
    

    # CREATE NETWORK MODEL
    TIME_PERIODS=ntimepts
    print(ntimepts)
    num_classes = 1 #0 or 1 for binary classification; if more desired, loss function must be changed
    BATCH_SIZE = 300
    EPOCHS = 7 # this can be changed if convergence occurs quickly
    
    model_m = Sequential()
    model_m.add(Conv1D(150, 4, activation='relu', input_shape=(TIME_PERIODS, 1)))
    model_m.add(Conv1D(200, 4, activation='relu'))
    model_m.add(MaxPooling1D(pool_size=3))                   #prevents overfitting
    model_m.add(Conv1D(50, 8, activation='relu'))
    model_m.add(Conv1D(25, 5, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
     #less sensitive to smalller variations by assigning 50% of data 0 weights
    layersize1 = 200
    layersize2 = 50
    if layersize1 != 0:
        model_m.add(Dense(units=layersize1))  # Hidden layer
    if layersize2 != 0:
        model_m.add(Dense(units=layersize2))  # Hidden layer 2
    model_m.add(Dense(num_classes))
    model_m.add(Activation('sigmoid'))
    #print(model_m.summary()) 
    
    print("Training " + netname +" with "+str(model_m.count_params())+" params.")
    model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])


    #  TRAIN MODEL
    trainingFiles = []
    for file in os.listdir(traindir):  # pull all training files from directory
        if file.endswith(".mat"):
            trainingFiles.append(file)

    nsets = len(trainingFiles)  # number of training files
    for j in range(nsets):  # loop over the different training files
        f_waves = h5py.File(traindir + '/' + trainingFiles[j],mode='r')  # load training file

        waveforms = np.transpose(f_waves['waveData'][()])
        x_train = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
        y_train = waveforms[:, 0]  # waveform labels
        x_train = np.expand_dims(x_train, axis=2)
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))

        then  = datetime.now() 
        model_m.fit(x_train,
                              y_train,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_split=0.2,
                              verbose=1)

    print('...................finished training.........................')
    now  = datetime.now()
    duration = now - then                         # For build-in functions
    duration_in_s = duration.total_seconds()      # Total number of seconds between dates
    minutes = divmod(duration_in_s, 60)[0]
    
    print("Model "+str(netname) +" trained in "+str(minutes) +"min.")
    # evaluate model
    testingFiles = []
    for file in os.listdir(testdir):  # pull all training files from directory
        if file.endswith(".mat"):
            testingFiles.append(file)
            
    nsets = len(testingFiles)  # number of training files
    testacc=[]
    print('Testing on ' + str(nsets) + ' files.')
    for j in range(nsets):  # loop over the different testing files
        f_waves = h5py.File(testdir + '/' + testingFiles[j],mode='r')  # load testing file(s)

        waveforms = np.transpose(f_waves['waveData'][()])
        x_test = waveforms[:, 1:(ntimepts+1)]  # waveform voltage values
        y_test = waveforms[:, 0]  # waveform labels
        x_test = np.expand_dims(x_test, axis=2)
        _, accuracy = model_m.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        accuracy = round(accuracy, 3)
        testacc.append(accuracy)
        print(testingFiles[j] + ' had a batch accuracy of ' +str(accuracy*100)+ '% with '+ netname)

    print(str(netname) + " had a combined acc of: " +str(np.mean(testacc)) + " with " +str(model_m.count_params()) + " params.")
    
    # SAVE PARAMETERS, by default saved in working folder
    #savefolder = '/Models/'+netname
    model_m.save(netname)  # save Keras model
    
    return

    # setrandomseed() ensures all necessary libraries are using a pre-set random seed
    # instructions from https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['PYTHONHASHSEED'] = str(0)

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seednum)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(seednum)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.compat.v2.random.set_seed(seednum)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    return
