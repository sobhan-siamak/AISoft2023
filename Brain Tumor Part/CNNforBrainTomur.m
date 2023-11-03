%% Workshop
% @copy by Hamed Sadravi - present in aisoft 2023
% CNN classifier for Brain Tomur 
close all
clear
clc

%% Load and Explore Image Data
BrainTomurDatasetPath = fullfile('F:\workshop\BrainData');

imds = imageDatastore(BrainTomurDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Display some of the images in the datastore
figure;
perm = randperm(80,15);
for i = 1:15
    subplot(3,5,i);
    imshow(imds.Files{perm(i)});
end

%Calculate the number of images in each category
labelCount = countEachLabel(imds)

%% specify the size of the images in the input layer of the network.
img = readimage(imds,1);
size(img)

%% Specify Training and Validation Sets
numTrainFiles = 30;   %75
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Define Network Architecture
layers = [
    imageInputLayer([227   227     3])
    
    convolution2dLayer(3,16,'Padding','same') %8
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same') %16
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same') %32
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',25, ...
    'MiniBatchSize', 16, ... 
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train Network Using Training Data
network = trainNetwork(imdsTrain,layers,options);

%% Classify Validation Images and Compute Accuracy
YPred = classify(network,imdsTrain);
YTrain = imdsTrain.Labels;
accTrain = sum(YPred == YTrain)/numel(YTrain)

YPred = classify(network,imdsValidation);
YValidation = imdsValidation.Labels;
accVal = sum(YPred == YValidation)/numel(YValidation)

%% Save Network
save('F:\workshop\BrainNetwork&Detector\networkCNNForBrain','network','layers'); 
