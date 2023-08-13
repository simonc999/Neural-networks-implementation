%% Hopfield - Riconoscimento di Template

clear 
close all
clc

% Riconoscimento delle lettere tramite rete di Hopfield, rete ideale per il riconoscimento di pattern all'interno di dati rumorosi

% function letterHop()

% vedi script SOM in cui è descritto il dataset
letters = prprob;

% 7x5=35 quindi non possiamo archiviare correttamente 26 attrattori che
% sono troppi rispetto al numero di neuroni che abbiamo a disposizione:
% dobbiam sceglierne un po' meno
memletters = 5;
template = letters(:,1:memletters);

% la rete di Hopfield è codificata con 1 e -1 per cui dove abbiamo degli
% zeri dobiamo sostituire con -1
template(template==0) = -1;

% costruzione della rete e memorizzazione degli attrattori
net = newhop(template);

% per usare la rete di Hopfield devo impostare il dato rumoroso come stato
% iniziale 'Ai' - statp iniziale da cui poi faccio evolvere in modo
% autonomo la rete
% h = net(~,~,Ai);

% creo inizialmente il dato rumoroso aggiungendo rumore ai dati
% in ogni lettera sporchiamo il 20% dei pixel
noisypx = round(0.2*size(letters,1));   % quanti pixel rumorosi
% noisypx = round(0.3*size(letters,1)); 
% noisypx = round(0.4*size(letters,1));
randix = randperm(size(letters,1));     % randomizzazione indici dei pixel rumorosi
noisyix = randix(1:noisypx);            % indici dei pixel rumorosi

% scegliamo una lettera di test da rendere rumorosa pervedere come si
% comporterà la rete 
letix = ceil(rand(1)*memletters);  % ceil arrotonda evitando di includere lo zero
noisylet = letters(:,letix);       % lettera originale scelta casualmente
                                   % la prendo da letters che ha codifica 0-1
                                                          
% la lettera originale la sporchiamo andando a negare i bit che avevamo
% scelto a caso prima come pixel che dovevano essere rumorosi; sono gli
% indici rumorosi di noisylet che devono essere negati
noisylet(noisyix) = double(not(noisylet(noisyix)));
noisylet(noisylet==0) = -1;       % scrupolo



% numero di step per i quali la faccio evolvere, passi che fa la rete in
% modo autonomo
nstep = 100;

% simulo la rete per questi nstep passi a partire da noisylet che diventa
% lo stato iniziale Ai che imponiamo alla rete di Hopfield - attenzione che
% uso af (e non y) che è lo stato per vedere i risultati della rete
[y, pf, af] = net({1 nstep},{},noisylet);

% rimetto template in 0-1 così posso usare not
% uso una nuova variabile per non sovrascrivere 
forma = template;
forma(forma == -1) = 0;
rumorosa = noisylet;
rumorosa(rumorosa == -1) = 0;
netstate = af{1};
netstate(netstate==-1) = 0;

% -----------------------------------------------%
% rappresentazione grafica
figure('Name','Hopfield')
subplot(1,3,1)
imshow(reshape(not(forma(:,letix)),5,7)')
title('Lettera originale')

subplot(1,3,2)
imshow(reshape(not(rumorosa),5,7)')
title('Lettera rumorosa')

subplot(1,3,3)
imshow(reshape(not(netstate),5,7)')
title('Lettera ricostruita')

% end 

