function classMotionSense(data)
    % Classify sequences read from MotionSense dataset files
    % Check dataset numerosities
    
    summary(data.ydata)
    
    %Prepare dataset
    [trainInd, valInd, testInd] = dividerand(numel(data.ydata), 0.7, 0.15, 0.15);
    
    xtrain = data.xdata(trainInd);
    ytrain = data.ydata(trainInd);
    
    xval = data.xdata(valInd);
    yval = data.ydata(valInd);
    
    xtest = data.xdata(testInd);
    ytest = data.ydata(testInd);
    
    %% Networ specs 
    
    %Hyperparameters 
    numHiddenunits = 200;
    numClasses = numel(categories(data.ydata));
    miniBatchSize = 64;
    maxEpochs = 30;
    numFeatures = size(data.xdata{1}, 1);
    
    % Define network
    layers = [ ...
        sequenceInputLayer(numFeatures);
        convolution1dLayer(16, numFeatures, 'Padding', 'same')
        maxPooling1dLayer(4, 'Padding', 'same')
        reluLayer()
        dropoutLayer(0.4)
        lstmLayer(numHiddenunits, 'OutputMode', 'last') 
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
        ];
    
    % Define options
    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
       'MiniBatchSize', miniBatchSize , ...
        'ValidationData', {xval, yval}, ...
        'InitialLearnRate', 0.008, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', maxEpochs/10, ...
        'LearnRateDropFactor', 0.9, ...
        'Verbose' , 0, ...
        'Plots','training-progress') % set minimo di opzioni
        
        net = trainNetwork(xtrain, ytrain, layers, options);
        
        
    % Verify on test set
    ypred = classify(net, xtest);
    
    acc = sum(ypred == ytest)./numel(ytest);
   
end