% Esercitazione 8.1

function out=featSelGA(popsize,ngen,pcross,pmut,data)

opengl('software') % (NB: serve a me)

% data --> è la struttura con training e testset create con prepareCADds
% funzione per la selezione delle feature per GA
% per l'ottimizzazione di un classificatore

% Serve per richiamare le funzioni salvate
addpath 'C:\Users\amind\OneDrive\Desktop\Operatori GA\operators'

% Inizializziamo popolazione iniziale
% i cromosomi di solito per colonna
pop=round(rand(height(data.trData),popsize));

% Fitness
fitvect=popFit(pop,data);

for i=1:ngen*popsize/2
    [strvec,strix]=sort(fitvect);
    ix1=roulettewheel(fitvect);
    ix2=roulettewheel(fitvect);

    if rand(1)<=pcross
        offs=twop_cross(pop(:,ix1),pop(:,ix2));
    else
        offs=[pop(:,ix1) pop(:,ix2)];
    end

    % Mutazione
    offs=bin_mut(offs,pmut);
    newfit=popFit(offs,data);
    [ordfit,ordix]=sort(newfit);

    % Se un cromosoma ha fitness migliore di almeno un elemento nella
    % popolazione, lo sostituiamo all'elemento peggiore

    if ordfit(end)>=strvec(1)
        pop(:,strix(1))=offs(:,ordix(end));
        fitvect(strix(1))=ordfit(end);     % Aggiorno vettore fitness
    end
    
    % Evoluzione GA
    maxfit(i,1)=max(fitvect);
    meanfit(i,1)=mean(fitvect);

end

% Visualizzo evoluzione GA
figure()
plot([maxfit,meanfit])
title('Evoluzione del GA');
legend('Max fit','Mean fit');

% Miglior cromosoma finale
[m,ix]=max(fitvect);
Cromosoma_Migliore=pop(:,ix);

% Fitness function
function out=popFit(pop,data)

% trClass=ind2vec(data.trClass);
for i=1:size(pop,2)
    ixsel=logical(pop(:,i));
    tdata=data.trData(ixsel,:);

    % Costruisco una rete con 1 strato nascosto con 5 neuroni
    net=patternnet(5);
    net=train(net,tdata,data.trClass);

    tsdata=data.tsData(ixsel,:);
    y=net(tsdata);

    % Accuratezza regolarizzata con numero di attributi
    outi(i)=sum((data.tsClass)==y)/length(data.tsData) - 0.01*sum(pop(:,i));
end

% COMMENTO: featSelGA(10,10,0.7,0.05,data)
