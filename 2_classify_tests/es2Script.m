%% ESERCITAZIONE 2
% script di supporto alla funzione netClassify
% clear all
clear
close all
clc

% ----------------------------------------------------------------------- %
% leggere e visualizzare il dataset
% ha tre colonne: x1, x2, classe
load dataset1.mat
    
figure('Name','Visualizzazione dei dati')
plot(data(data(:,3)==1,1),data(data(:,3)==1,2),'ro', ...
     data(data(:,3)==2,1),data(data(:,3)==2,2),'sb')

% ho diverse configurazioni di reti - case
 net5 = netClassify(5);
% net1 = netClassify(1);
% net2 = netClassify(2);
% net = netClassify(3);
% net = netClassify(4);

% visualizzazione della rete e caratteristiche principali
% view(net)

% for i=1:4
%     figure('Name','Errori compiuti dalla rete')
%     netClassify(i)
%     subplot(4,1,i)
%     plot(xtest(1,erryt~=0),xtest(1,erryt~=0),'*k')
%     hold on
% 
% end