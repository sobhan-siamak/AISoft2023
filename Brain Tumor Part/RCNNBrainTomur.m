%% Workshop
% @copy by Hamed Sadravi - present in aisoft 2023
%%%%      Train R-CNN Brain Tomur Detector   
clc
clear
close all
%% Apply network and Detection training command
TrainDetector=0; % 1:Train and save Detector 0:load and use Detector 
n=5;             % Photo number to Show Lable
perm =8;         % Photo number to test the Detector
mod='all'; % 'all' 'max'

%% load Network 
load('F:\workshop\BrainNetwork&Detector\networkCNNForBrain')

%% Load Images and Lables
load('F:\workshop\Data&Image\BrainTomurTable1st.mat')
imdsDetector = imageDatastore(BrainTomurTable1st.imageFilename);
bldsDetector = boxLabelDatastore(BrainTomurTable1st(:,2:end));
ds = combine(imdsDetector, bldsDetector);

%% Display one of the images in the imdsDetector
ShowLableF(imdsDetector,bldsDetector,n)

%% Options for trayning Object Detections 
InitialLearnRate=0.001; 
MaxEpochs= 6;  %Number of training rounds
MiniBatchSize=16;%Size of the mini-batch to use for each training iteration
Shuffle='every-epoch'; % 'once' 'never' 'every-epoch'
optionsOD = TrainOptionsODF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);

%% Train Object Detections
if TrainDetector==1
 RCNNDetector = trainRCNNObjectDetector(BrainTomurTable1st, ...
        network, optionsOD);

 %'NegativeOverlapRange', [0 0.3] , ... 
 %       'PositiveOverlapRange', [0.7 1]

 % save Detector
 save('F:\workshop\Net&Detector\rcnnODBrainTomur' ,'RCNNDetector','imdsDetector','bldsDetector')
else
 % load Detector
 load('F:\workshop\Net&Detector\rcnnODBrainTomur')
end

%% Evaluate RCNNDetector 
EvaluateF(RCNNDetector,imdsDetector,bldsDetector);

%% Test RCNNDetector with Image
TestDetectorF(imdsDetector,RCNNDetector,perm,mod)
