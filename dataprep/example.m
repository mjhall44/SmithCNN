clear all; close all; 

% don't forget to add folders to run path
% change the ~ below to the package save location
inputDir = 'Z:\SmithLab\prjNAS\NASNet\Package\dataprep\presorted\'; %location of hand labeled nevs and mats to combine for training
outputDir = 'Z:\SmithLab\prjNAS\NASNet\Package\';
numSplits = 1;

createNetworkFiles(inputDir,outputDir, numSplits)
% creates a conglomerate of all nevs and mats within the input director
% folder structure should be:
% inputDir   : where all nevs and mats are in subfolders
% --subfolder: named with channels desired from nevs eg. "1-96"
%              or any name if containing .mats
% numSplits  : splits waveforms the into this number of training files
%              if using a large amount of data on one machine, 2-3 may train without
%              exceeding memory
% outputDir  : places a split of training and testing data into two folders
%              within this directory
%              "training_files" and "testing_files" folders will be created


outputDir = 'Z:\SmithLab\prjNAS\NASNet\Package\test_mini\';
createFilesforClassification(inputDir,outputDir)
% used to load any files, and saves them with space for labels
% inputDir   :
% --subfolder: named with channels desired from nevs eg. "1-96"
%              or any name if containing .mats
% outputDir  :places .mats of input data into this folder, best used for
%             further classification