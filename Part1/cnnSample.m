% cnnPreprocess();

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

% obsolete
% dataTrainstoreSubset = imageDatastore('..\');
% dataTeststoreSubset = imageDatastore('..\');

dataTrainstoreSubset.Files = dataTrainstore.Files;
dataTeststoreSubset.Files = dataTeststore.Files;

dataTrainstoreSubset.Labels = dataTrainstore.Labels;
dataTeststoreSubset.Labels = dataTeststore.Labels;

imageDim = 28;

layers = [...
    imageInputLayer([imageDim imageDim])
	 convolution2dLayer([9, 9],20)
	 averagePooling2dLayer([2, 2])
%      convolution2dLayer([5, 5],50), ...
%      averagePooling2dLayer([2, 2]), ...
     fullyConnectedLayer(10)
	 softmaxLayer()
     classificationLayer()];
 
 options = trainingOptions('sgdm', ... 
            'MaxEpochs', 3,...
            'InitialLearnRate',0.001, ...
            'L2Regularization', 0.001, ...
            'Momentum', 0.9, ...
            'MiniBatchSize', 500 ...
        );

disp('Training...');
[convnet, history] = trainNetwork(dataTrainstoreSubset,layers,options);
plotperform(history);

disp('Training Done...');

disp('Testing...');
YTest = classify(convnet, dataTeststoreSubset);
TTest = dataTeststoreSubset.Labels;
accuracy = sum(YTest == TTest)/numel(YTest);
disp(accuracy);