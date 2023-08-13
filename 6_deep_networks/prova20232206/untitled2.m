%% TEMA ESAME - ECG da classificare con rete CNN
% Ho un dataset di 188 colonne di cui: 
% 187 rappresentamo un battito
% la 188-esima è la classe: normal/abnormal 0/1

close all
clear 
clc

% ---------------------------------------------------- %
% leggo i dataset
ab = readtable("ptbdb_abnormal.csv"); % abnormal
no = readtable("ptbdb_normal.csv");   % normal
AB = table2array(ab);
NO = table2array(no);

abnormal = AB(1:10,1:(end-1));
normal = NO(1:10,1:(end-1));

figure('Name','Grafico iniziale')
subplot(2,2,1)
plot(1:size(NO,2)-1,normal,'b')
title('Normal')

subplot(2,2,2)
plot(1:size(AB,2)-1,abnormal,'r')
title('Abnormal')

subplot(2,2,3)
plot(1:size(NO,2)-1,NO(1,1:(end-1)),'b-')
hold on
plot(1:size(AB,2)-1,AB(15,1:(end-1)),'r-')
legend('normal','abnormal')
title('Confronto')

% creo il dataset  shuffled
T = [AB; NO];
dataset = T(randperm(size(T,1),size(T,1)),:);


% ----------------------------------------------------------------------- %
% CREAZIONE dei DATASET: TRAIN - VALIATION - TEST 
% ----------------------------------------------------------------------- %
% divisione training (70%) - val (15%) - test (15%) 

ixtrain = round(0.7*size(dataset,1));
ixtest = ixtrain+((size(dataset,1)-ixtrain)/2);

trainSet = dataset(1:ixtrain,:);
valSet = dataset(ixtrain+1:ixtest,:);
testSet = dataset(ixtest+1:end,:);

% [trainInd,valInd,testInd] = dividerand(size(dataset,1),0.7,0.15,0.15);
% trainSet = dataset(trainInd,:);
% valSet = dataset(valInd,:);
% testSet = dataset(testInd,:);


%% ----------------------------------------------------------------------- %
% TRASFORMAZIONE DEI DATASET
% ----------------------------------------------------------------------- %

% TRAINING
% per allenare la rete il training set deve essere in forma di cell
% sono 10186 celle ognuna contenente il vettore di battiti da 187 elementi
xtrain = con2seq(trainSet(:,1:end-1)')';

% il target deve essere un cell array di dati categorici
classe = trainSet(:,end)';          % prendo la colonna della classe
ytrain = cell(size(classe,2),1);    % preistanzio il cell
for i = 1:numel(classe)
    ytrain{i} = categorical(classe(i));
end

% -----------%
% VALIDATION
% nelle options il validation lo vuole sotto forma di cell array di due
% elementi di cui uno è formato dai dati e il seconodo è un cell array
% della classe in formato categorico
xval = con2seq(valSet(:,1:end-1)')';
valclass = valSet(:,end)';                 % prendo la colonna della classe
yval = cell(size(valclass,2),1);           % preistanzio il cell

for i = 1:numel(valclass) 
    yval{i} = categorical(valclass(i)); % target
end

validation = cell(1,2);
validation{1} = xval;
validation{2} = yval;

% -----------%
% TEST
xtest = con2seq(testSet(:,1:end-1)')'; % esempi senza classe
tclasse = testSet(:,end)';             % prendo la colonna della classe
ytest = cell(size(tclasse,2),1);       % preistanzio il cell
for i = 1:numel(tclasse)
    ytest{i} = categorical(tclasse(i));
end


% ---------------------------------------------------- %
% alla rete devo passare il numero delle classi
nClass = numel(unique(dataset(:,end)));

% dimensione del ostro vettore
inputDim = size(dataset,2)-1;


% ----------------------------------------------------------------------- %
% COSTRUZIONE LAYER BY LAYER
% ----------------------------------------------------------------------- %
layers = [

    sequenceInputLayer(inputDim)

    convolution1dLayer(1,8,'padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(1)

    convolution1dLayer(1,16,'padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(1)

    convolution1dLayer(1,32,'padding','same')
    batchNormalizationLayer
    reluLayer
    
    % parte di classificazione 
    % dropoutLayer
    fullyConnectedLayer(nClass)
    softmaxLayer 
    classificationLayer
 ];


% ----------------------------------------------------------------------- %
% OPTIONS di Training
% ----------------------------------------------------------------------- %
% imposto le options per fare training
options = trainingOptions( ...
    'sgdm',...
    'InitialLearnRate',0.01,...
    'LearnRateDropFactor',0.002,...
    'MaxEpochs',10,...                % 100
    'Shuffle','every-epoch',...
    'ValidationData',validation,...
    'ValidationFrequency',5,...       % 20
    'Verbose',false,...               % metto falso per sopprimerli
    'Plots','training-progress');


% ----------------------------------------------------------------------- %
% ALLENAMENTO DELLA RETE
% ----------------------------------------------------------------------- %
% alleno la rete 
net = trainNetwork(xtrain,ytrain,layers,options);


%% ----------------------------------------------------------------------- %
% ACCURATEZZA - MATRICE DI CONFUSIONE
% ----------------------------------------------------------------------- %
% prestazioni sul test set
YPred = classify(net,xtest);
somma = 0;
ypred = zeros(size(YPred,1),1);

for i =1:size(YPred,1)
    add = (YPred{i}==ytest{i});
    somma = somma+add;
    ypred(i) = YPred{i};
end 

ypred(ypred == 2) = 0;
ypred = logical(ypred)';
yTest = logical(testSet(:,end))';

% accuratezza
testAcc = somma/numel(ytest);
fprintf('Accuratezza sul test set %2.2f \n', testAcc);

% matrice di confusione
figure
plotconfusion(yTest,ypred)


%% ----------------------------------------------------------------------- %
% SALVATAGGIO .MAT
% ----------------------------------------------------------------------- %
% Salvataggio della rete neurale in un file .mat

File = 'Salvataggio.mat';
save(File, 'net','xtrain','ytrain','xval','yval','validation','xtest','ytest');                               % rete

%% ----------------------------------------------------------------------- %
% INFORMAZIONI SUL MIO DATASET
% ----------------------------------------------------------------------- %
% percentuali di zeri e uni nei diversi set

p_uni_train = (numel(find(trainSet(:,end))))*100/(size(trainSet,1));
p_zeri_train = (numel(find(trainSet(:,end) == 0)))*100/(size(trainSet,1));

p_uni_val = (numel(find(valSet(:,end))))*100/(size(valSet,1));
p_zeri_val = (numel(find(valSet(:,end) == 0)))*100/(size(valSet,1));

p_uni_test = (numel(find(testSet(:,end))))*100/(size(testSet,1));
p_zeri_test = (numel(find(testSet(:,end) == 0)))*100/(size(testSet,1));

disp('PERCENTUALI CHE DEFINISCONO IL DATASET OTTENUTO')
fprintf('Il training set è formato dal %2.2f perc di dati in classe Normal \n', p_zeri_train);
fprintf('Il training set è formato dal %2.2f perc di dati in classe Abnormal \n\n', p_uni_train);

fprintf('Il validation set è formato dal %2.2f perc di dati in classe Normal \n', p_zeri_val);
fprintf('Il validation set è formato dal %2.2f perc di dati in classe Abnormal \n\n', p_uni_val);

fprintf('Il test set è formato dal %2.2f perc di dati in classe Normal \n', p_zeri_test);
fprintf('Il test set è formato dal %2.2f perc di dati in classe Abnormal \n\n', p_uni_test);

% pulizia
clear p_uni_train p_zeri_train p_uni_val p_zeri_val
clear p_uni_test p_zeri_test ab no AB NO normal abnormal