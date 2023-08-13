%function  CNNdigits()
clc
clear
close all
% matlabroot troviamo una seria di sottocartelle
     % nella cartella toolbox troviamo nnet che contiene una serie di
     % sottocartelle legate a specifiche app, nella cartella nndemos
     % troviamo nndataset dove ci sono datasets per esercitarsi con le reti
     % tra cui la cartella DigitDataset con ogni cartella una serie di immagini a
     % livelli di grigio

     % carichiamo quindi le 10 cartelle in un digitDatastore presente nella
     % cartella prima

     % RECAP
     % IN MATLAB\R2021b\toolbox\nnet\nndemos\nndatasets ABBIAMO LE CARTELLE
     % DI IMMAGINI

     % IN MATLAB\R2021b\toolbox\nnet\nndemos\nndatasets
     % ABBIAMO DIGITDATASTORE
imds = digitDatastore;
img = readimage(imds,10);

% teniamo traccia della dimensione dell'immagine

imgSize = size(img);

% vediamo che imds ha delle proprieta'
% ------ imds.countEachLabel
% ci restituirà numero di elementi a video per ognuna delle etichette
% ans =
% 
%   10×2 table
% 
%     Label    Count
%     _____    _____
% 
%       0      1000 
%       1      1000 
%       2      1000 
%       3      1000 
%       4      1000 
%       5      1000 
%       6      1000 
%       7      1000 
%       8      1000 
%       9      1000 

lCount= imds.countEachLabel;

% se vogliamo organmizzarci il dataset con training e set dovremmo
% randomizzare l'ordine dei nostri elementi

% il totale sarà size(imds.Files)
% quindi noi faremo

% rndix = randperm(numel(imds.Files));

% conterra' da 1 a 10k randomizzati

% per fare la suddivisione stratificata usiamo splitEachLabel
% che prende un oggetto imds in input e andrà ad estrarre in modo
% stratificato i nostri esempi restituendo 2 img datastore in uscita: uno
% di dimensione P (se gli diamo .7 conterrà 70% degli esempi stratificati e
% bilanciati) l'altro di .3


% al posto di usare randperm usiamo randomized integrata nella funzione
[trainDS,tmpDS] = splitEachLabel(imds,.7,'randomized');



[valDS,testDS] = splitEachLabel(tmpDS,0.5,'randomized');

% quindi cosi abbiamo train validation e test
% visualizzo alcuni esempi
rndidx = randperm(numel((trainDS.Files)));
figure
for i = 1:9
    subplot(3,3,i)
    imshow(trainDS.readimage(rndidx(i)))
end

% andaiamo ora a costruire la rete convoluzionale per strati

inputSize = [imgSize 1]; % 1 perchè abbiamo un solo canale (bianco e nero e non RGB)

% dobbiamo quindi fornire il numero delle classi che possiamo vedere come
% la prima colonna di lCount

% questo perchè ponendo parametri generalizzati non inseriamo parametri
% numerici (eccccccipiaceeee :))))))

numClasses = size(lCount,1);

% possiamo quindi costruire la nostra rete inizializzando una variabile
% layers che è un vettore di strati che specifica in cascata da sx a dx il
% tipo di strato che vogliamo utilizzare 

% c'e' un tool deepNetworkDesigner che è un tool grafico che è utile per
% costruire una rete 
% abbiamo degli strati di ingresso in particolare i primi due sono usati
% per preparare delle immagini per essere poi ingressi di una rete
% imageInputLayer è quello utilizzato
% sequence... è per le serie temporali

% in giallo abbiamo le diverse convoluzioni in modo 1 2 o 3 dimensionali
% e convoluzioni trasposte che servono tipicamente come autoencoder nella
% parte di modifica

% GUARDA RETEAMANO.png
% abbiamo da settare nel primo layer 28x28x1 essendo le immagini di
% dimensione 28x28 e in bianco e nero

% si può fare anche qua 
layers = [ 
    imageInputLayer(inputSize)
    convolution2dLayer(3,6,'Padding','same') % 6 filtri 
    batchNormalizationLayer
    reluLayer % per buttare via tutto cio che è negativo e avere una derivata paria  1
    fullyConnectedLayer(numClasses) % deve sapere quante classi ci sono
    softmaxLayer % probabilità che il determinato in ingresso appartenga a quella classe
    classificationLayer
    ];

% abbiamo così definito gli strati che compongono questa rete
% dovremmo modificare il training col training options definendo le
% caratteristiche 
% la prima opzione è il tipo di algoritmo di apprendiemnto
% SGDM sta per stocastic radiant descending method

% Learn Ratio Drop Factor = 0.2 diminuisce il tasso di apprendiemnto del
% 20% ad ogni giro

% ---------------------------------------------------------------------------%
% iperparametri di guida del processo di apprendimento
% abbiamo così definito gli strati che compongono questa rete
% dovremmo modificare il training col training options:
% --> 'sgdm' - il tipo di algoritmo di apprendiemnto SGDM sta per stocastic radiant 
%     descending method - discesa del gradiente stocastica
% --> 'InitialLearnRate': Tasso di apprendimento iniziale
% --> 'Learn Rate Drop Factor' = 0.2 diminuisce il tasso di apprendiemnto del
%     20% ad ogni giro, qua metto meno perche parto da 0.01
% --> 'MaxEpochs': limito il numero di epoche di apprendimento a 10
% --> 'Shuffle': posso chiedere alla rete di fare uno shuffling dei dati
%     posso mettere 'ones' 'never' 'every-epoch'
% --> la validation frequency va settata in accordo col numero massimo di
%     epoche che impongo, se metto 100 epoche devo diminuire la frequenza per i
%     costi computazionali
% --> plots - voglio come feedback come evolve il training, ma potrei
%     vedere anche come evolve sulla validation

% --> minibatchsize - usato nelle LSTM (non qua) definisce la dimensione
%     di un insieme di esempi che costituiscono quasi un nuovo concetto di
%     epoca, l'aggiornamento dei pesi sinaptici avviene alla fine della
%     presentazione di un batch; sottoepoca nella quale modificare i parametri: ogni 
%     64 esempi viene fatta una valutazione della funzione di costo che si applica 
%     alla back propagation andando a modificare i nostri pesi sinaptici

% options = trainingOptions( ...
%     'sgdm',...
%     'InitialLearnRate',0.01,...
%     'LearnRateDropFactor',0.002,...
%     'MaxEpochs',10,...
%     'Shuffle','every-epoch',...
%     'ValidationData',imdsVal,...
%     'ValidationFrequency',5,...
%     'Verbose',false,...                   % metto falso per sopprimerli
%     'Plots','training-progress');
% 
options = trainingOptions('sgdm',... % usiamo il mertodo del gradiente stocastico 
    'InitialLearnRate',0.01,...
    'LearnRateDropFactor',0.002,...
    'MaxEpochs',10,... % limitando il numero di epoche di allenamento a 4 
    'ValidationData',valDS,... % impostiamo il validation dataset
    'Shuffle','every-epoch',...
    'ValidationFrequency',5,... % 2 volte ogni epoca 10/5.settiamo la frequenza di validazione a 5. A seconda della dimensione del dataset dobbiamo considerare l'onere: se il numero fosse 100 possiamo fare ogni 20
    'Verbose',false, ... % si può definire che tipo di informazioni avere nella command window
    'Plots','training-progress'); % e vogliamo tenere traccia del training progress parametro di allenamento

net = trainNetwork(trainDS,layers,options);

% GUARDA OUPUT.png

YPred = classify(net,testDS);
YTest = testDS.Labels;

testAcc = sum(YPred==YTest)/numel(YTest);

fprintf('Accuratezza sul test set %2.2f\n', testAcc);

% l'imageInput Layer prende come parametro indispensabile la dimensione
% dell'ingresso 28 28 1 poi accetta una serie di coppie valore valore
% centra il valore a 0. Quindi i nostri dati interi nellapprendimento sono
% dati normalizzati, e per utilizzare altre opzioni bisogna specificare i
% parametri ad esempio min e max


% le prestazioni sono già ottime ma se vogliamo utilizzare la rete meglio
% per un problema più generale e potenziare quindi la rete
% possiamo ridefinire gli strati

layers = [
    imageInputLayer([imgSize 1]) % si ha di solito un aumento del numero di filtri quindi partiamo con un 2d
    convolution2dLayer(3,8,'Padding','same')% mettiamo 8 filtri e specifichiamo come gestire la dimensione dell'immagine col padding che ci mantenga l'immagine invariata
    % padding same per non modificare la dimensione originale (non voglio
    % che le uscite vengano trimmate ) 
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2) % va a ridurre la dimensione dell'immagine facendo uno stride pari a 2
        % ripetiamo quindi i blocchi

     convolution2dLayer(3,16,'Padding','same')%la dimensione è stata ridotta dal pooling
    batchNormalizationLayer
    reluLayer
     maxPooling2dLayer(2,'Stride',2)
     convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer;
    ];

net = trainNetwork(trainDS,layers,options);

% GUARDA OUPUT.png

YPred = classify(net,testDS);
YTest = testDS.Labels;

testAcc = sum(YPred==YTest)/numel(YTest);
deepNetworkDesigner(net)
%end

