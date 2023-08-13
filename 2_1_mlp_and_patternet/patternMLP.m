
%% CONFRONTO MLP e Patternet

function patternMLP()

% Usuamo il dataset sulle cellule cqncerogene, sono 699 pz. con 9 attributi
% distinti in due classi
    %   1. Benign
    %   2. Malignant
load cancer_dataset.mat

dImputs = cancerInputs;
dTarget = cancerTargets;

% ------------------------------------------------------------ %
% classificazione con MLP - strato in uscita lineare

% iniziamo con 8 neuroni 
% net = feedforwardnet(8);

% posso anche pensare di usare lo stesso numero di neuroni ma
% distribuendoli su piu strati
% net = feedforwardnet([6 2]);
net = feedforwardnet([4 2]);


% se lo runno piu volte, e vedo che statisticamente le performance sono le
% stesse anche se diminuisco il numero di neuroni allora devo prefereire un numero basso di neuroni

% ------------------------------------------------------------ %
% divisione training - validation - test
% uso 'dividerand' che prende come argomento il numero di istanze del
% datasete e le proporzioni di dati che voglio ottenere, mi restituisce gli
% indici con cui suddividee
[trainInd, valInd, testInd] = dividerand(size(dImputs,2),0.8,0,0.2);

% diamo zero al set validation, poi con 'divideparam' andiamo a dividere il
% trining set in train e validation
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

% --------------------------------------------------------------------- %
% trasformo il target set in classi che i fatto sono gli indici dei vettori
% colonna che rappresentano questo tipo di classificazione, le
% uscitevengono trasformate in un vettore 1xn in cui la classe è codificata
% in modo numerico
indTarget = vec2ind(dTarget);

% --------------------------------------------------------------------- %
% allenamento della rete
net = train(net,dImputs(:,trainInd),indTarget(:,trainInd));
% view(net)

% --------------------------------------------------------------------- %
% Performance sul test set
% errori di classificazione sull'arrotondamento del valore della classe 
y = net(dImputs(:,testInd));
err = abs(indTarget(:,testInd)-round(y)); 

% ho due classi quindi l'errore non può essere maggiore di uno ma per
% essere generici lo prendo come bouleano nel caso in cui usassi lo script
% per un dataset con piu classi
perf = sum(err>0)/size(y,2);

% --------------------------------------------------------------------- %
fprintf('La rete MLP commette %d errori su %d ingressi, pari a %2.2f \n',...
    sum(err>0),size(y,2),perf);

% --------------------------------------------------------------------- %
% matrice di confusione
% per farla devo ritornare alla rappresentazione precedente quindi non in
% indici ma con il numero corrispondente
precedente = ind2vec(round(y));

figure(1)
plotconfusion(dTarget(:,testInd),precedente);

% gcf = get current figure, mi serve per prendere la handle della figura
% corrente - le figure sono oggetti con handle che hanno una serie di
% proprietà e contenuti, la figura è la scatola, se metto degli assi,
% questi assi sono un altro oggetto ancora 
set(gcf,'Name','MLP');


% --------------------------------------------------------------------- %
% --------------------------------------------------------------------- %
% Patternet - strato d'uscita competitivo
% anche in questo caso i target glieli devo passare in forma indiciale
% pnet = patternnet(hiddenSize, trainFcn, performFcn);
% 'trainFcn' = tipo di funzione di apprendimento
% 'performFcn' = calcolo dell'errore del funzionale di costo (default = cross entropia)
pnet = patternnet([4 2]);
pnet.divideParam.trainRatio = 0.8;
pnet.divideParam.valRatio = 0.2;
pnet.divideParam.testRatio = 0;

% training 
pnet = train(pnet,dImputs(:,trainInd),dTarget(:,trainInd));
view(pnet)

% calcolo dell'errore su codifica numerica della classificazione
% !! attenzione che devo richiamare la net giusta !!
py = pnet(dImputs(:,testInd));
perr = abs(indTarget(:,testInd)-vec2ind(py));
pperf = sum(perr>0)/size(py,2);

% --------------------------------------------------------------------- %
fprintf('La rete PATTRNET commette %d errori su %d ingressi, pari a %2.2f \n',...
    sum(perr>0),size(py,2),pperf);

figure(2) 
plotconfusion(dTarget(:,testInd),py);
set(gcf,'Name','PATTERNET');

end 
