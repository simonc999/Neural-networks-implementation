%% SISTEMA DINAMICO - RETE RICORRENTE 

% Usiamo una rete ricorrente di tipo layer recurrent net per apprendere il
% comportamento di un sistema dinamico costruendo un training set di
% comportamenti ingresso-uscita di un modello rappresentato da una funzione
% di trasferimento, ad esempio:
%             (1+s)             zero in 1    [rad/s] [1 1]  [1 E 1 CHE MOLTIPLICA S]
% G(s) = --------------- -->    polo in 0.2  [rad/s]
%        (5s+1)(0.2s+1)         polo in 5    [rad/s]

% ---------------------------------------------------------------------- %

%function net = dynsys(nn, net)
clc
clear
close all
% prende in ingresso un certo numero dineurone ed eventualmente la rete
% stessa nel caso si voglia simulare o 'riapprendere'/ proseguire
% l'apprendiemento della rete

%if nargin<1
    nn = 10; % default neuron hidden layer
%end 

% ------------------------------------------------------------------- %
% costruzione della funzione di trasferimento
num = [1 1];                % zero (termine di grado 0 e termine di grado 1)
den = conv([5 1],[0.2 1]);  % 'prodotto' dei poli 
sys = tf(num,den);          % tf: transfr function
%bode(sys)

%%
% ------------------------------------------------------------------- %
% costruzione del training set
Tc = 0.05;                       % discretizzazione
t = 0:Tc:30;                     % vettore dei tempi (sono 601=(30/0.05)+1 sec)

% Abbiamovisto che l'ingresso che abbiamo messo (u) non va bene: 
% Nonostante il training set sia perfettamente simulato, l’ingresso di test (generalizzazione) mai visto nella fase di training non viene riprodotto adeguatamente, allora modifichiamo u
% u = [zeros(1,20) ones(1,581)];  % ingresso
u0 = [zeros(1,20) ones(1,100) zeros(1,200) ones(1,80) zeros(1,80) 0:Tc:1];   
u1 = [sin(2*pi*t(1:100))];
u = [u0 u1];                      % ingresso

y = lsim(sys,u,t);                % risposta del sistema

fh = figure('Name','Training set');
plot(t,u,t,y)
legend('Ingresso','Ripsosta')
% ciò non è sufficiente a far apprendere correttamente la rete

% ------------------------------------------------------------------- %
% Trasformiamo il training set in sequenze: ogni elemento diventa una
% cell - passo dalla rappresentazione a 'vettore' che viene considerato
% 'concorrente' quindi viene rappresentato tutto in una sola volta, ad una
% sequenza in cui ogni elemento è un ingresso a sè stante
us = con2seq(u);  
ys = con2seq(y');  % target della rete


% ------------------------------------------------------------------- %
% Istanziamo la rete (che viene inizializzata vuota):
% net = layrecnet(layerDelays,hiddenSizes,Fcn);
% 'layerDelays': vettore di ritardi, ho l'uscita dello strato nescosto 
%               retroazionata sull'ingresso con dei ritardi lungo l'anello 
%               di retroazione
% 'hiddenSizes': è il nostro 'nn'

net = layrecnet([1:2],nn);
% view(net)
%%
% ------------------------------------------------------------------- %
% Preparazione di ingressi che devono essere sfalsati per gestire
% i ritardi e target - nel nostro caso non ho ritardi sull'ingresso ma ho 2 ritardi sull'ingresso dello strato nascosto - Ai contiene il vettore che
% mi consente di inizializzare gli stati degli hidden layers per l'inizio
% della simulazione della rete
% [] = preparets(net,Xnf,Tnf) con Xnf=us e Tnf=ys
[Xs,Xi,Ai,Ys] = preparets(net,us,ys);

% Ai sono gli stati
% Xs — Shifted inputs
% 
% Xi — Initial input delay states
% 
% Ai — Initial layer delay states
% 
% Ys — Shifted targets

% Di default il numero delle epoche è 1000, se non è troppo lento posso
% asciare a 1000 altrimenti riduco
% net.trainParam.epochs = 500;    % numero epoche

% ------------------------------------------------------------------- %
% training
net = train(net,Xs,Ys,Xi,Ai);
yns = net(us);
yn = seq2con(yns);      % la ritrasformo da sequenza a vettore


% ------------------------------------------------------------------- %
% grafico 
figure(fh)
hold on
plot(t,yn{1},'m--')
legend('input','sys out', 'net out')


% ----------------------------------------------------------------------- %
% verifica generalizzazione - test set
% Magari la rete si comporta bene come risposta al gradino, ma noi vogliamo verificare il comportamento anche a fronte di un altro ingresso -

% ut = [zeros(1,20) 2*sin(2*pi*4*t(1:301)) zeros(1,280)];
% abbasso la frequenza da 4Hz a 0.2Hz per vedere meglio il comportamento
ut = [zeros(1,20) 2*sin(2*pi*0.2*t(1:301)) zeros(1,280)]; 

yt1 = lsim(sys,ut,t);
uts = con2seq(ut);

% simuliamo
nsout = net(uts);
testout = seq2con(nsout); % torniamo a vettore

figure('Name','Generalizzazione')
plot(t,ut,t,yt1,t,testout{1})
legend('input','sys out', 'net out')

%end 
