clear all; close all; 

% don't forget to add folders to run path
% change the ~ below to the package save location

inputDir = 'C:\~\SmithCNN-1.0\train_mini\'; %location of hand labeled nevs and mats to combine for training
outputDir = 'C:\~\SmithCNN-1.0\';
numSplits = 1;
trainandtest = 1;

createNetworkFiles(inputDir,outputDir, numSplits, trainandtest)
% creates a conglomerate of all nevs and mats within the input director
% folder structure should be:
% inputDir   : where all nevs and mats ***must*** be in subfolders
% --subfolder: named with channels desired from nevs eg. "1-96"
%              or any name if containing .mats
% outputDir  : places a split of training and/or testing data into two folders
%              within this directory
%              "training_files" and "testing_files" folders will be created
% numSplits  : splits waveforms into this number of training files and/or testing files
%              if using a large amount of data on one machine, 2-3 may train without
%              exceeding memory
% trainandtest : 1 / creates both training and testing (validation) files
%                0 / creates only training data


outputDir = 'Z:\SmithLab\prjNAS\NASNet\Package\test_mini\';
createFilesforClassification(inputDir,outputDir)
% used to load any files, and saves them with space for labels
% inputDir   :
% --subfolder: named with channels desired from nevs eg. "1-96"
%              or any name if containing .mats
% outputDir  : places .mats of input data into this folder, best used for
%              further classification
