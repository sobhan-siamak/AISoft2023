%% Workshop
% @copy by Hamed Sadravi - present in aisoft 2023
%%%%      Train Faster R-CNN Brain Tomur Detector   
clc
clear
close all
%% Apply network and Detection training command
TrainDetector=0;  % 1:Train and save Detector 0:load and use Detector 
n=1;              % Photo number to Show Lable
perm =12;          % Photo number to test the Detector
mod='all';        % 'all' 'max'

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
InitialLearnRate=0.01;  %1=0.001
MaxEpochs= 6;    %Number of training rounds
MiniBatchSize=1;  % for FastRCNN and FasterRCNN only 1 is Correct 
Shuffle='every-epoch';   % 'once' 'never' 'every-epoch'
optionsOD = TrainOptionsODF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);

%% Train Object Detections
if TrainDetector==1
 tic
 FasterRCNNDetector = trainFasterRCNNObjectDetector(BrainTomurTable1st, ...
        network, optionsOD);

%, 'NegativeOverlapRange', [0 0.3] , ... 
%        'PositiveOverlapRange', [0.5 1]

 toc
 % save Detector
 save('F:\workshop\Net&Detector\fasterrcnnODBrain' ,'FasterRCNNDetector','imdsDetector','bldsDetector')
else
 % load Detector
 load('F:\workshop\Net&Detector\fasterrcnnODBrain')
end

%% Evaluate FasterRCNN Detector 
results=EvaluateF(FasterRCNNDetector,imdsDetector,bldsDetector);

%% Test FasterRCNN Detector with Image
TestDetectorF(imdsDetector,FasterRCNNDetector,perm,mod)
