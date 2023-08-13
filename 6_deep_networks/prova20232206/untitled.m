%%
clc
clear
close all
T_abnormal = readtable("ptbdb_abnormal.csv");
T_normal = readtable("ptbdb_normal.csv");


plot_test_table_ab = table2array(T_abnormal(1:10,1:(end-1)));
plot_test_table_no = table2array(T_normal(1:10,1:(end-1)));

figure()
subplot(2,1,1)
plot(1:size(plot_test_table_no,2),plot_test_table_ab,'r')
subplot(2,1,2)
plot(1:size(plot_test_table_no,2),plot_test_table_no,'b')



T= [T_abnormal;T_normal];

T_shuffle = T(randperm(size(T,1)),:);
array = table2array(T_shuffle);


inputSize = size(array,2)-1;


numClasses = numel(unique(array(:,end)));

[trainIND,valIND,testIND] = dividerand(size(array,1),0.7,0.15,0.15);
trainSet = array(trainIND,:);
testSet = array(testIND,:);
valSet = array(valIND,:);

train_perc0 = sum(trainSet(:,end)==0)/(size(trainSet,1));
test_perc0 = sum(testSet(:,end)==0)/(size(testSet,1));
val_perc0 = sum(valSet(:,end)==0)/(size(valSet,1));

% ----------------------------------------------------------------------- %
% TRASFORMAZIONE DEI DATASET
% ----------------------------------------------------------------------- %

% TRAINING
% per allenare la rete il training set deve essere in forma di cell
% sono 10186 celle ognuna contenente il vettore di battiti da 187 elementi
xtrain = con2seq(trainSet(:,1:end-1)');

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
% alla rete devo passare il numero delle classi
nClass = numel(unique(array(:,end)));

% dimensione del ostro vettore
inputDim = size(array,2)-1;




layers = [ 
    sequenceInputLayer(inputSize,"Name","layer 1")
    convolution1dLayer(1,8,'Padding','same',"Name","layer 2")
    batchNormalizationLayer("Name","layer 3")
    reluLayer("Name","layer 4")
    maxPooling1dLayer(1,"Name","layer 5")
    convolution1dLayer(1,16,'Padding','same',"Name","layer 6")
    batchNormalizationLayer("Name","layer 7")
    reluLayer("Name","layer 8")
    maxPooling1dLayer(1,"Name","layer 9")
    convolution1dLayer(1,32,'Padding','same',"Name","layer 10")
    batchNormalizationLayer("Name","layer 11")
    reluLayer("Name","layer 12")
    maxPooling1dLayer(1,"Name","layer 13")
    fullyConnectedLayer(numClasses,"Name","layer 14") % deve sapere quante classi ci sono
    softmaxLayer("Name","layer 15") % probabilità che il determinato in ingresso appartenga a quella classe
    classificationLayer("Name","layer 16")
    ];


options = trainingOptions('sgdm',... % usiamo il mertodo del gradiente stocastico 
    'InitialLearnRate',0.01,...
    'LearnRateDropFactor',0.002,...
    'ValidationData',validation,...
    'MaxEpochs',2,... % limitando il numero di epoche di allenamento a 4 
    'Shuffle','every-epoch',...
    'ValidationFrequency',5,... % settiamo la frequenza di validazione a 5. A seconda della dimensione del dataset dobbiamo considerare l'onere: se il numero fosse 100 possiamo fare ogni 20 (ogni 20 epoche fa la validazione)
    'Verbose',false, ... % si può definire che tipo di informazioni avere nella command window
    'Plots','training-progress'); % e vogliamo tenere traccia del training progress parametro di allenamento

net = trainNetwork(xtrain,ytrain,layers,options);


% ----------------------------------------------------------------------- %
% ACCURATEZZA 
% ----------------------------------------------------------------------- %
% prestazioni sul test set
YPred = classify(net,xtest);

for i=1:numel(YPred)
    A(i)=YPred{i}==categorical(1);
end

somma = 0;

for i =1:size(YPred,1)
   
    add = (YPred{i}==ytest{i});
    somma = somma+add;
end 

figure()
confusionchart(logical(testSet(:,end))',A)
testAcc = somma/numel(ytest);

fprintf('Accuratezza sul test set %2.2f \n', testAcc);
s.a = net;
s.b = T;