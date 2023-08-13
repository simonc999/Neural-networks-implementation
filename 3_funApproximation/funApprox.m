function net = funApprox(nType,tSize)
rng(12345) % blocco il seed

% utilizziamo ora mlp o grnn per approssimare le funzioni
% Function approximation with MLP o GRNN
% nType = 1 MLP
% nTYpe = 2 GRNN
if nargin<1
    nType=1;
end
if nargin<2
    tSize = 0.2;
end
% prendiamo somma di sinusoidi con coefficienti casuali

% definiamo il tempo in cui la definiamo
t = -5:0.01:5;

% generiamo un vettore casuale di valori
a = rand(1,9)*2; % 2hz
% noi stiamo campionando a 100 hz e quindi rispettiamo il teorema di
% shannon
% e il valore della funzione che sarà quindi 
y = a(1)*cos(2*pi*a(2)*t + a(3)) + ...
    a(4)*cos(2*pi*a(5)*t + a(6)) + ...
    a(7)*cos(2*pi*a(8)*t + a(9));
fh=figure();
plot(t,y)

% SECONDA SIMULAZIONE
tsamp = 1:5:numel(t);
xtrain = t(tsamp);
ttrain = y(tsamp);

% t vettore ordinato
ntest= round(tSize*size(t',1));
ntrain=size(t',1)-ntest;

% creiamo il vettore di indici
% randperm costruisce una permutazione di numeri casuali sulla dimensione
% dei tempi

randix = randperm(size(t',1));

% ne prendiamo un numero pari al numero del train
ixtrain = randix(1:ntrain);
ixtest = randix(ntrain +1 : end);

% xtrain = t(ixtrain);
% ttrain = y(ixtrain);
xtest = t(ixtest);
ttest = y(ixtest);


% definisco lo spread per grnn
spread = 0.5;
% SECONDA SIMULAZIONE
spread = 0.02;
spread = 0.05;

% effettuiamo uno shuffling del test e del training
switch nType
    case 1
        % costruiamo la rete sempre con due strati nascosti di 5 neuroni e
        % 2 neuroni
        % net = feedforwardnet([5 2]);
        % SECONDA SIMULAZIONE
        net = feedforwardnet([20 2]);
        net = feedforwardnet([20 4]);
        net.divideFcn = '';
        % SECONDA SIMULAZIONE COMMENTO SOTTO
%         net.divideParam.testRatio=0;
%         net.divideParam.trainRatio=0.85;
%         net.divideParam.valRatio=0.15;
        net = train(net,xtrain,ttrain);
    case 2 % GRNN
        % ha una sintassi simile alla rbf
        net = newgrnn(xtrain,ttrain,spread);
end
% per vedere come si comporta complessivamente la rete possiamo passare
% l'intero vettore dei tempi
yn = net(t);
figure(fh)
hold on 

plot(xtrain,ttrain,'ok')
plot(t,yn,'r')
% con la prima simulazione il risultato è scarso

% un altro approccio che possiamo considerare è quello di passare alla rete
% per il training una versione sottocampionata e quindi costruisco il
% nostro training set usando un passo diverso e passare dalla frequenza di
% campionamento di 100 hz anche a meno della metà

% SECONDA SIMULAZIONE 
% con lo spread impostato (correlato al tempo di campionamento) i nostri
% esempi di training sono distanti tra loro 50 ms 
% con un piccolo spread vogliamo avere gaussiane sovrapposte 

% rimodulo ancora: impongo spread = 0.05 in modo che la gaussiana abbia
% dimensione tra i nostri esempi

% SECONDA MLP


% le due reti vedono entrambe il raining set
% ma hanno apporcci diverse
% la grnn usa apporssimazioni locali basati su centri delle gaussiane su
% esempi disponibili sul training set
% abbiamo quindi sotto campionamento regolare e i dati che prima erano
% rappresentati a 10 ora sono a 50 ms, ogni 50 ms la grnn posiziona una
% gaussiana che servirà da base per approssimazione locale in quell'intorno
% con una sovrapposzizione tra diverse gaussiane che viene risolta in una
% produzione dell'uscita sulla base dei pesi dell'ultimo strato

% nel mlp dobbiamo costruire approx attraverso modello non su approx locale
% ma generale e in questo caso dipenderà dai GDL che consentiamo alla rete
% quindi connessioni sinapriche e quanto potrà adattarsi al nostro training

% ottenendo una buona approssimazione del grnn abbiamo visto che i gdl
% della rete era 201 pesi sinaptici , quindi per gfare confronto di un mlp
% abbiamo consentito a mlp un numero confrontabile di parametri liberi
% aumentando i neuroni nei due strati nascosti con cui abbiamo organizzato
% i neuroni

% quindi da una parte gioco su spread dall'altra sui numeri dei neuroni.


% per fare un confronto equo dobbiamo bloccare gli indici con un settaggio
% di seed