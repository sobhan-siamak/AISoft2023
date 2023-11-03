%% Workshop
% @copy by Hamed Sadravi - present in aisoft 2023
%%%%      Train Fast R-CNN Brain Tomur Detector   
clc
clear
close all
%% Apply network and Detection training command
TrainDetector=1; % 1:Train and save Detector 0:load and use Detector 
n=1;             % Photo number to Show Lable
perm =10;         % Photo number to test the Detector
mod='max';       % 'all' 'max'

%% load Network 
load('F:\workshop\BrainNetwork&Detector\networkCNNForBrain') 

%% Load Images and Lables
load('F:\workshop\Data&Image\BrainTomurTable1st')
imdsDetector = imageDatastore(BrainTomurTable1st.imageFilename);
bldsDetector = boxLabelDatastore(BrainTomurTable1st(:,2:end));
ds = combine(imdsDetector, bldsDetector);

%% Display one of the images in the imdsDetector
ShowLableF(imdsDetector,bldsDetector,n)

%% Options for trayning Object Detections 
InitialLearnRate=0.0001;  
MaxEpochs= 10;   %Number of training rounds
MiniBatchSize=1; % for FastRCNN and FasterRCNN only 1 is Correct 
Shuffle='every-epoch'; % 'once' 'never' 'every-epoch'
optionsOD = TrainOptionsODF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);

%% Train Object Detections
if TrainDetector==1
 FastRCNNDetector = trainFastRCNNObjectDetector(BrainTomurTable1st, ...
        network, optionsOD);

%, 'NegativeOverlapRange', [0 0.3] , ... 
%        'PositiveOverlapRange', [0.5 1]

 % save Detector
 save('F:\workshop\Net&Detector\fastrcnnODBrain' ,'FastRCNNDetector','imdsDetector','bldsDetector')
else
 % load Detector
 load('F:\workshop\Net&Detector\fastrcnnODBrain')
end

%% Evaluate FastRCNN Detector 
EvaluateF(FastRCNNDetector,imdsDetector,bldsDetector);

%% Test FastRCNN Detector with Image
TestDetectorF(imdsDetector,FastRCNNDetector,perm,mod)
