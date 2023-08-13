%function  cluster = irisSom()
% la prima som più semplice può essere utilizzato per il clustering
% dell'ires dataset
clear 
clc
load fisheriris.mat
data = iris_dataset;
X = meas(:,1:2);
% la funzione prende le dimensioni del nostro lattice, se diamo un solo
% numero facciamo lattice monodimensionale di 5 ordini, se invece diamo
% vettore di due elementi avremo lattice bi-dimensionale con un numero di
% elementi pari al prodotto dei due elementi
% con 2 2 avrò 4 neuroni

net = selforgmap([2 2]);
net = train(net,data);
% possiamo quindi simulare la rete sul dataset

out = net(data);

% per ognuno degli ingressi avremo una colonna di 4 valori in cui 3 sono
% settati a 0 e uno è quello vincitore
% per trovare il cluster usiamo

cluster = vec2ind(out);
gscatter(X(:,1),X(:,2),species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
hold on
for i=1:150
    if(cluster(i)==1)
        color = 'red';
    elseif(cluster(i)==2)
            color = 'green';
    else
        color = 'blue';
    end
    scatter(X(i,1),X(i,2),5,color,'filled')
    hold on
end

% i pulsanti sotto servono per analizzare

% SOM topology ci mostra il layout del lattice bi dimensionale
% è disponibile gia' prima dell'allenamento

% SOM sample hits per ognuno dei 4 neuroni ci dice quante volte ha vinto

% SOM Neib conn ci indicano le distanze e le connessioni, le distanze sono
% tutte pari a 1 tranne i due neuroni estremi

% SOM weight position ci fa vedere come sono organizzati i vari neuroni al
% termine dell'apprendiemnto. 

% SOM neig weig dist ci fa vedere la disposiione dei neuroni nello spazio
% dei pesi. La codifica va dal nero al giallo per indicare dal più vicino
% al più lontano dei pesi tra i neuroni vicini del lattice

% l'uscita sarà una codifica dei nostri 150 ingressi ciscuno un numero
% intero che ci dice quale dei 4 neuroni risponde (vince) per
% quell'ingresso. Quindi in quale delle 4 classi viene classificato il
% singolo ingresso




%end

