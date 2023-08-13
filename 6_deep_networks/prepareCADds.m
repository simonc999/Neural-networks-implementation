% Esercitazione 8

% GA che fa una feature selection, scegliere quali feature scegliere per la
% classificazione

% L'idea: uso algoritmo per selezione degli attributi
% Tanti geni quanti sono gli attributi
% codifica binaria: 1 usiamo attributo, 0 non lo usiamo

% Nel datset si hanno attributi di diverso tipo
% La rete vuole un ingresso riga

function out=prepareCADds

% Lettura dei dati
data=readtable('C:\Users\amind\OneDrive\Desktop\Operatori GA\CAD.xlsx');

% Rappresentazione numerica dei dati
% CAD --->1
% Norm --->0
class=vec2ind(onehotencode(categorical(data.Cath),2)');

% Sovrascrivo i dati della tabella, devo lasciare i dati in formato colonna
data.Sex=vec2ind(onehotencode(categorical(data.Sex),2)')';
data.Obesity=vec2ind(onehotencode(categorical(data.Obesity),2)')';
data.SystolicMurmur=vec2ind(onehotencode(categorical(data.SystolicMurmur),2)')';
data.DiastolicMurmur=vec2ind(onehotencode(categorical(data.DiastolicMurmur),2)')';
data.Dyspnea=vec2ind(onehotencode(categorical(data.Dyspnea),2)')';
data.Nonanginal=vec2ind(onehotencode(categorical(data.Nonanginal),2)')';
data.LowTHAng=vec2ind(onehotencode(categorical(data.LowTHAng),2)')';
data.ThyroidDisease=vec2ind(onehotencode(categorical(data.ThyroidDisease),2)')';
data.Atypical=vec2ind(onehotencode(categorical(data.Atypical),2)')';
data.ExertionalCP=vec2ind(onehotencode(categorical(data.ExertionalCP),2)')';
data.BBB=vec2ind(onehotencode(categorical(data.BBB),2)')';
data.VHD=vec2ind(onehotencode(categorical(data.VHD),2)')';
data.LVH=vec2ind(onehotencode(categorical(data.LVH),2)')';

% Attributi che scartiamo
data.CRF=[];
data.CVA=[];
data.AirwayDisease=[];
data.CHF=[];
data.DLP=[];
data.WeakPeripheralPulse=[];
data.LungRales=[];
data.PoorRProgression=[];
data.Cath=[];

data=table2array(data);

% Preparo Dataset: Train e Test

[trainInd,valInd,testInd]=dividerand(height(data),0.8,0,0.2);

out.trData=data(trainInd,:)';
out.trClass=class(trainInd);
out.tsData=data(testInd,:)';
out.tsClass=class(testInd);

% Possiamo fare girare qualsiasi rete

% Penso che abbia fatto cosi
% net=patternnet(5);
% view(net)


