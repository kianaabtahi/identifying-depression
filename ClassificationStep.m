
clc; clear; close all
%================================Necessary Paths===========================

CurrentPath = pwd;
addpath (genpath(CurrentPath))

%=================================Intializing==============================
%% Initializing parameters

Num_CVKfold_FS = 10;
PercentOfSelectedFeat = 50; % As on percentage
GAParam.nPop = 50;
GAParam.MaxItration = 15;
instanceDataAugmentation = 1;
featuresDataAugmentation = 1;

%==========================2D-Normalization================================

load Processed

Data=[A(2:end,2:end-1);B(2:end,2:end-1);C(2:end,2:end-1)];
Label=[A(2:end,end);B(2:end,end);C(2:end,end)];

%===============================Features selection===========================
   
Class = zeros(size(Label,1),1);

for p = 1:size(Label,1)
    for q = 1 : size(Label,2)
    if Label(p,q)<9.1
        Class(p,q) = 1;
    elseif Label(p,q)>8.99 & Label(p,q)<11.99
        Class(p,q) = 2;
    elseif  Label(p,q)>11.99 & Label(p,q)<14.99
        Class(p,q) = 3;
     elseif  Label(p,q)>14.99
        Class(p,q) = 4;
    else
        t(p,q) = p;
    end
    end
end

%% Loading Data
disp('..........................................................................................................................')

[convertedData,convertedLabel,ClassNames] = DataLoading(Data,Class,instanceDataAugmentation,featuresDataAugmentation);

%% Data mixing 
randomLocations = randperm(numel(Label));
ChangeTargets = convertedLabel(randomLocations); % A 1-D list at this point
ChangeData = convertedData(:,:,randomLocations);

%% Initiaizing cnn
h = size(ChangeData,1);
w = size(ChangeData,2);
no_of_epochs = 5;
batch_size = 3;
dataDivisionRatio = 0.3;
NumKfold = 10;
Features = reshape(ChangeData,[h*w size(ChangeData,3)]);


[trainCNN,testCNN] = crossvalind('holdout',ChangeTargets,dataDivisionRatio);
 
    [TrainData,TrainClass,TestData,TestClass] = DividingData(Features,ChangeTargets,trainCNN,testCNN,batch_size);
    
    train_x = reshape(TrainData,h, w,[]);
    train_y = BuildingLabel(TrainClass);  
    test_x = reshape(TestData,h, w,[]); 
    test_y = BuildingLabel(TestClass); 
    cnn = TrainingCNNClassifier(train_x, train_y, ChangeTargets, no_of_epochs, batch_size,h, w);
    
    %% Train reults
    [TrainConfMatrix, TrainErr,PredictedTrainLabel] = testcnn(cnn, train_x,train_y);
    figure, opt.mode='both';plotConfMat(TrainConfMatrix, ClassNames,opt);
    
    %% Test reults
    [TestConfMatrix, TestErr,PredictedTestLabel] = testcnn(cnn, test_x, test_y);
    figure, opt.mode='both';plotConfMat(TestConfMatrix, ClassNames,opt);

save('Results','cnn','PredictedTrainLabel','PredictedTestLabel')



