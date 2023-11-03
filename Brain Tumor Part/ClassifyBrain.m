%% CNN classifier for Brain Tomur 
% @copy by Hamed Sadravi - present in aisoft 2023

close all
clear
clc

%% Load Trained Network
load('F:\workshop\BrainNetwork&Detector\networkCNNForBrain2nds')

%% Read and Resize Image
% directory to read images : D:\Datasets
I = imread('F:\workshop\unknown\unknown1.jpeg'); % unknown2.jpg
figure
imshow(I)
size(I);  %N.N

% I = imresize(I,inputSize);
% figure
% imshow(I)

classNames = network.Layers(end).Classes; % N.N
numClasses = numel(classNames);              % N.N

%% Classify Image
[label,scores] = classify(network,I);
figure
 imshow(I)
  title (string(label) + ", " + num2str(100*scores(label),3) + "%");

%% Display Top Predictions (N.N)
[~,idx] = sort(scores,'descend'); %Descending column numbers based on cell data
idx = idx(2:-1:1); % five predicted 
classNamesTop = network.Layers(end).Classes(idx);
scoresTop = scores(idx); % five predicted scores  
figure
 barh(scoresTop)
  xlim([0 1])
  xlabel('\fontsize{14} \fontname{Times New Roman}Probability')
  yticklabels(classNamesTop)
  title('\fontsize{16} \fontname{Times New Roman} Top 5 Predictions')