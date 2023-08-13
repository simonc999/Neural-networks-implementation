%function cluster = letterSOM()
% ora chiediamo alla som di clusterizzare una codifica  binaria delle
% lettere dell'alfabeto organizzate in matrici 7x5 di pixel.
% per esempio (brutto)
% quindi a --0--
%          -0--0
%          -0000
%          0---0

% usiamo il dataset Dataset prprob salvato in kiro
close all
clear 
clc
letters = prprob;

% per vedere come si presenta uno dei caratteri uso la funzione reshape che
% ci sonsente di trasformare matrici vettori cambiando di dimensione 

subplot(1,2,1),imshow(reshape(not(letters(:,1)),5,7)');


% inizialiazziamo quindi la nostra rete
% possiamo in modo analogo chiedere alla rete di costruire 4 classi

net = selforgmap([3 2]);
net = train(net,letters);

out = net(letters);
% facciamo ora un ciclo per raccogliere tutte le lettere riconosciute dal
% primo neurone dal secondo etc
% per ognuno dei neuroni andiamo a visualizzare con dei subplot le lettere
% andando ad associare anche le lettere dell'alfabeto
% facciamo quindi una variabile che contiene dei template in  termini di
% caratteri 

alfabeto = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

% ciclando quindi sul numero di righe del nostro out  

for i=1:size(out,1)
    % diremo quali sono le lettere identificate dai diversi neuroni nel
    % cluster ci mettiamo tutte le lettere dell'alfabeto che vanno a finire
    % in quel cluster

    cluster{i}=alfabeto(out(i,:)==1);

    % l'indice delle lettere riconosciuto 
    lettix{i}=find(out(i,:)==1);

    % useremo quindi questi per mettere in una figura con tanti subplot
    % quante sono le lettere riconosciute
    ncluster = numel(lettix{i});
    figure()
    for j=1:ncluster
        subplot(1,ncluster,j)
        imshow(reshape(not(letters(:,lettix{i}(j))),5,7)');
    end
     

end

%end

