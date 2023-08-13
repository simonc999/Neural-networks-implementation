%% FUNZIONE DI CLASSIFICAZIONE
% alla funzione devo dare in ingresso il tipo di rete e grandezza

function net = netClassify(nType,tSize)
% neural network classifier
% nType =1 MLP; =2 RBF esatta; n=3 RBF approssimata; n=4 PNN (probabilistica) ; n=5 patternet
% che è un MLP con uno strato di uscita competitivo
% tSize porzione del training da usare come test set
% close all
%clear


if nargin<1
    nType=5;
end
if nargin<2
    tSize = 0.2;
end
% ------------per le RBF c'è da inizializzare il parametro di spread
% spread = 0.1; % for RBF layers
% il criterio empirico è definire uno spread che copre almeno la metà
% dell'estensione dello spazio degli ingressi ma dipende anche dal numero
% di esempi e dall'architettura che si sceglie di utilizzare
% in una architettura esatta e approssimata cambia il risultato perchè la
% sovrapposizione delle gaussiane diventa anche un limite perchè è un
% problema sovradimensionato. Utilizzando un numero piccolo di neuroni si
% può iniziare impostando lo spread = 1
spread = 0.07;
spread = 0.05;
% ------------per le RBF approssimata 
errRBFApp = 0.1;
errRBFApp = 0.01;
neuroniInputRBFApp=400;

% impostazione manuale del seed per bloccare sequenza generata
rng(12345)

 load('dataset1.mat');
%load('dataset2.mat'); 
% con il dataset 2 abbiamo forme più fittabili con gaussiane
% abbiamo coord x1 coord x2 e classificazione

fh=figure
plot(data(data(:,3)==1,1),data(data(:,3)==1,2),'ro',...
    data(data(:,3)==2,1),data(data(:,3)==2,2),'sg')

% la regione sul confine è difficile da considerare

ntest= round(tSize*size(data,1));
ntrain=size(data,1)-ntest;

% i dati in questo caso sono già mischiati in classificazione altrimenti
% bisognava generare un numero di indici casuali

% randperm(10);

% genera numeri da 1 a 10 senza ripetizione

xtrain = data(1:ntrain,1:2)';

% TRASPOSTO: i tool vogliono un esempio per colonna quindi bisogna trasporre le prime
% due colonne perchè ogni esempio deve essere unja colonna quindi dovrà
% essere 2xN

% il target sarà sulla terza colonna


ttrain=data(1:ntrain,3)';

xtest = data(ntrain+1:end,1:2)';
ttest = data(ntrain+1:end,3)';

% iniziamo quindi a  costruire e ad allenare la rete

switch nType
    case 1 % MLP 
        % costruiamo il primo oggetto rete

        % prende in imput il numero di neuroni degli strati nascosti/
        % nel nostro caso il primo con 5 neuroni e il secondo con 2
        % questa istruzione istanzia l'oggetto rete posizionando il numero
        % di neuroni ma senza costruire ancora nessuno strato di uscita

        % la rete poi viene configurata con il configure() o viene
        % configurata automaticamente sul training
        % abbiamo due strati nascosti: nel primo abbiamo 5 neuroni e nel
        % secondo ne abbiamo 2
        % l'uscita è rappresentata dallo strato di uscita costituita d a 3
        % neuroni

        net=feedforwardnet([5 2]);

        % chiamiamo il training (allenamento della rete)
        % con gli ingressi di training sia i target consente alla rete di
        % configurarsi, xtrain è un vettore in cui ogni colonna rappresenta
        % una tupla di ingresso
        % ttraining rappresenta l'uscita (target)
        % si può configurare la rete senza allenarla con configure che ha
        % bisogno degli stessi ingressi
        % divideran: divide automaticamente la rete usa:
        % 70 training - 15 validation e 15 test
        % parametri modificabili che quinid andiamo a modificare
        net.divideParam.testRatio=0;
        net.divideParam.trainRatio=0.85;
        net.divideParam.valRatio=0.15;

        net = train(net,xtrain,ttrain);

        % con net.layers

        % con il dataset 2 confronta perfettamente senza errori
        % avrebbe senso valutarne una cross validazione
    case 2 % RBF esatta, ci aspettiamo un neurone per ogni esempio
        net = newrbe(xtrain,ttrain,spread);
        % vengono automaticamente posizionati i neruoni del training set e
        % non abbiamo un passaggio di allenamento e quindi possiamo
        % valutare direttamente il comportamento della rete: newrbe
        % posiziona e allena la rete

        % con dataset 2 ottengo un errore del 31.5 % con lo spread di prima
        % mettendo lo spread a 1 riduco l'errore a 0.75 %
    case 3 
        % qua invece c'è un processo di apprendimento
        net = newrb(xtrain,ttrain,errRBFApp,spread,neuroniInputRBFApp);
        % 0.01 è il goal
    case 4
        % ha come argoemtni gli stessi della rbf esatta ma ha una
        % peculiarità i target devono essere passati come indici
        ttrv = ind2vec(ttrain);
        net = newpnn(xtrain,ttrv,spread);
    case 5
        % la patternet è un MLP con uno strato di uscita con il numero di
        % neuroni pari al numero delle classi in modo che ogni calsse venga
        % identificata dal neurone
        % anche lui ha bisogno di convertire i target
        ttrv = ind2vec(ttrain);

        % il terzo argomento è la cross entropia è la valutazione della
        % performance della rete
        % è spesso più efficacie utilizzare un certo numero di neuroni
        % negli strati in cascata piuttosto che tutti nello stesso strato
        % questo patternet va allenato con train
        net = patternnet(5);

        net.divideParam.testRatio=0;
        net.divideParam.trainRatio=0.85;
        net.divideParam.valRatio=0.15;

        net = train(net,xtrain,ttrv);
        
        % anche qua possiamo impostare i poarametri su training validation
        % e test
        % per confrontare mlp e pattern avrebbe senso impostare il seed
        
end


ytest = net(xtest);

% riconverto 
if nType == 4 || nType ==5
    ytest = vec2ind(ytest);
end

% posizionandomi su ytest e facendo step vedo le uscite 
% ma vogliamo averle intere
% visualizzo gli errori sul test set: devo usare un arrotondamento
    % delle uscite in modo che possano esere confrontate con i nostri
    % target

erryt = ttest - round(ytest);
% aggiungiamo quindi al grafico gli errori non classificati correttamente
figure(fh)
hold on

plot(xtest(1,erryt~=0),xtest(2,erryt~=0),'*k')

% ci facciamo restituire prestazioni della rete 
errRatio = sum(abs(erryt))/numel(ytest);
fprintf('La rete commette il %2.2f percento di errori', errRatio*100)

% il risultato è diverso perchè dipendono dalle funzioni di
% inizializzazione, l'allenamento si può fermare in un punto ogni volta
% diverso
net = struct(net);
% per avere un numero comparabile dovrei settare il seed dei numeri casuali
% di matlab con rng(seed)
end






