function [map, best] = travelingGA(ncities,nchrom,pcross,pmut,map)
%% GA - Genetic Algorithm
rng(12345)
% Problema del commesso viaggiatore

% IL CAMION DEVE PASSARE DAPPERTUTTO E DEVE OTTIMIZZARE IL PERCORSO FACENDO
% MENO STRADA POSSIBILE TRA TUTTE LE CITTA'

% variante: la funzione di fitness non è solo in base alla distanza ma 
% anche al consumo che è influenzato da un aumento del carico del cammion. 
% Ogni quintale fa aumentare il consumo del 3%. 
% Il commesso va a prendere dei carichi nelle città e quindi nella mappa 
% oltre alla coordinata della città ci sarà anche un peso in quintali di 
% materiale che doveva prendere. Ogni volta che passa da una città aumenta 
% il carico totale e il consumo aumenta. Devo pensare ad una fitness da 
% aggiornare. Ogni segmento di strada viene pesato per il peso*(1+0.3).

% LA FITNESS SI BASA SUL FENOTIPO E NON SUL GENOTIPO, QUINDI DEVO FARLO
% ESPRIMERE PER VALUTARLA

%function map = travelingGA(ncities, nchrom, pcross, pmut, map)
clear 
close all
clc

% ncities = numero di città da visitare
% nchrom = numero di cromosomi
% pcross = probabilità di cross over
% pmut = prpbabilità di mutazione
% map = la mappa stessa nel caso la volessi aggiornare

if nargin<1
    ncities = 20;
end
if nargin<2
    nchrom = 50;
end 
if nargin<3
    pcross = 0.7;
end 
if nargin<4
    pmut = 0.05;
end 
% ncities = 20;
% nchrom = 50;
% pcross = 0.7;
% pmut = 0.05;

%
% proprio la costruzione della cartina: 
% è la fase in cui do delle coordinate alle diverse città
if nargin<5
    map = tsp_map(ncities);
end
%map = tsp_map(ncities);

% numero di generazioni
ngen = 10000;

% costruzione della popolazione iniziale
popsize = nchrom;

% crea una matrice dove ogni colonna è un ipotetico percorso per ordine di
% città

pop = tsp_pop(popsize,ncities);
%%
% --------------------------------------------------------------------- %
% FITNESS
% valutazione della fitness della popolazione, 'fitvec' è il vettore delle
% fitness degli elementi della popolazione nell'ordine in cui sono nella
% popolazione
fitvec = tsp_fit(pop,map);  % gli serve la mappa per calcolare le distanze 


% --------------------------------------------------------------------- %
% ELITISMO
% sostanzialmente devo mettere in ordine il vettore di fitness
[~, sortix] = sort(fitvec);



%%
% --------------------------------------------------------------------- %
% EVOLUZIONE
% iniziamo con l'evoluzione attraverso le generazioni
for i=1:ngen

    % inizializzo una nuova popolazione a matrice vuota
    newpop = [];

    % tengo traccia di cosa sta succedendo
    maxfit(i) = max(fitvec);       % idealmente non voglio che diminuisca
    meanfit(i) = mean(fitvec);
    
    % -------- %
    % elitismo
    % aggiungo i due migliori cromosomi che sono gli ultimi due perche la
    % sort ordina dal piu piccolo al piu grande
    newpop = [newpop pop(:,sortix(end-1:end))];


    % costruisco la nuova popolazione lavorando per coppie
    for j=1:(nchrom/2-1)  % -1 to adjust for elitism
        % sono 25 coppie -1 perchè una coppia viene considerata da prima
        % con elitismo
        
        % -------------------------------------------------------- %
        % TOURNAMENT
        % k è il parametro dell'algoritmo che fa decidere se prendermo il
        % cromosoma con la fitness + elevata nel caso si ricata in pk o
        % l'altro, mettendolo a 0.7 mi mantengo in una certa probabilità 
        % che fa sì che possa selezionare anche i cromosomi peggiori
        k = 0.7;
        chrINDEX1 = tournamentk(fitvec,k);
        chrINDEX2 = tournamentk(fitvec,k);
        
        % --------------------------------------------------------- %
        % CROSS OVER
        % tiriamo il dado per decidere se farlo o no 
        if rand(1) <= pcross
            % entrambi i discendenti vengono cambiati per cross over
            newchrom =  tsp_cross(pop(:,chrINDEX1),pop(:,chrINDEX2));
            % newchrom =  twop_cross(pop(:,chrINDEX1),pop(:,chrINDEX2));
        else 
            % se non faccio cross over rimangono i genitori
            newchrom = [pop(:,chrINDEX1) pop(:,chrINDEX2)];   
        end 
        
        % --------------------------------------------------------- %
        % MUTAZIONE
        newchrom = tsp_mut(newchrom,pmut);
	
	% esercizio: proviamo a farlo steady state:
    % dopo aver mutato devo calcolare la fitness e aggiungere il cromosoma ù
    % nella popolazione solo se la sua fitness è > della fitness minima; bisogna fare qualche piccola modifica. Inserisco i cromosomi che migliorano ed elimino quelli che hanno perso. Dal POV dell efficienza dovrei vedere dei miglioramenti
        
        % aggiornamento della popolazione
        newpop = [newpop newchrom];

    end 
    
    % ---------------------------------------------------------------- %
    % RICALCOLO FITNESS
    % siamo sempre nella i-esima popolazione, devo ricalcolare la nuova
    % fitness
    pop = newpop;
    fitvec = tsp_fit(pop,map);
    
    % ---------------------------------------------------------------- %
    % ELITISMO
    [~, sortix] = sort(fitvec);
end 




% ---------------------------------------------------------- %
% grafico 
% mostriamo la mappa percorsa secondo il miglior cromosoma
best = pop(:,sortix(end)); % miglior cromosoma

figure('Name','Best Chromosome')
plot(map(best,1), map(best,2),'b')
hold on
%figure('Name','Mappa')
scatter(map(:,1),map(:,2),map(:,3),'filled')
legend('citta','percorsp')
% --------------------------------------------------------------------- %
% funzione creazione mappa
% --------------------------------------------------------------------- %
    function map = tsp_map(ncities)

        % stiamo all'interno di un quadrato di 400km
        % tra -200 e + 200 per entrambe le dimensioni
        map_coord = rand(ncities,2)*400-200;
        map_weight = rand(ncities,1)*100;
        
        map = [map_coord map_weight];
        
        % disegnamo la mappa
        figure('Name','Mappa')
        % plot(map(:,1),map(:,2),'r+','MarkerSize',8)
        scatter(map(:,1),map(:,2),map(:,3),'filled')
        

    end 

%%
% --------------------------------------------------------------------- %
% funzione creazione popolazione
% --------------------------------------------------------------------- %
    function pop = tsp_pop(popsize,ncities)

        % creo la popolazione cromosomica come una permutazione di indici
        % di città
        for i=1:popsize
            pop(:,i) = randperm(ncities,ncities)'; % primio da che n a che n il secondo quantri generarne 
        end 
    end 


% --------------------------------------------------------------------- %
% funzione calcolo vettore di fitness
% --------------------------------------------------------------------- %
    function fitvec = tsp_fit(pop,map)
        
        % la fitness viene calcolata come 1 fratto la distanza complessiva 
        % del percorso
        for i =1:size(pop,2)

            % esercizio di astrazione e programmazione compatta: potremmo
            % specificare ogni tratto giocando sugli indici - cicliamo
            % sugli elementi di pop con due indici sfalsati di 1 rispetto
            % all'altro
            % xprimo = 0;
            % yprimo = 0;
            % ma per capirci un po' di più e per evitare errori io lo 
            % spezzo in parti - distanza euclidea
            x = map(pop(1:end-1,i),1) - map(pop(2:end,i),1);
            y = map(pop(1:end-1,i),2) - map(pop(2:end,i),2);
            z = map(pop(1:end,i),3);

            for j=1:numel(z)
                z_cumulative(j)= sum(z(1:j));
            end

            xprimo = map(pop(end,i),1)-map(pop(1,i),1);
            yprimo = map(pop(end,i),2)-map(pop(1,i),2);

            x = [x; xprimo];
            y = [y; yprimo];

            CONST_WEIGHTS = 0.03;
            CONST_DISTANCE = 1;

            weights_consume = sqrt(x.^2+y.^2).*z_cumulative'*CONST_WEIGHTS;
            distance_consume = sqrt(x.^2+y.^2)*CONST_DISTANCE;
            
           fitvec(i) = 1/(sum(weights_consume+distance_consume));
        end 

    end 


% --------------------------------------------------------------------- %
% funzione cross over
% --------------------------------------------------------------------- %
    function newchrom = tsp_cross(chr1, chr2)
        
        % tiro i dadi per calcolare l'indice da permutare
        % rmb. ceil arrotonda evitando lo zero
        ix = ceil(rand(1)*size(chr1,1));

        % per il primo tratto è uguale al genitore
        nc1 = chr1(1:ix);
        nc2 = chr2(1:ix);

       
        % dall'indice di permutazione in poi in nc1 aggiungo i pezzi di 
        % chr2 che ancora non sono in nc1 e iceversa aggiungo in nc2 i
        % pezzi di chr1 che ancora non sono in nc1

        % es. ix = 3    chr1 ----xxx  -->  nc1 ---ooxx
        %               chr2 xxoooxx  -->  nc2 xxo-xxx
        
        for i=1:size(chr1,1)
            if isempty(find(nc2==chr1(i)))
                nc2(end+1,1)=chr1(i);
            end
            if isempty(find(nc1==chr2(i)))
                nc1(end+1,1)=chr2(i);
            end
        end 

        % perche non facciamo semplicemente cosi?
        % nc1 = [chr1(1:ix) chr2(ix+1:end)];
        % nc2 = [chr2(1:ix) chr21(ix+1:end)];

        newchrom = [nc1 nc2];
    end 


% --------------------------------------------------------------------- %
% funzione mutazione
% --------------------------------------------------------------------- %
    function newchrom = tsp_mut(chrom, pmut)

        % Attenzione! è diverso dal caso del cross over in cui tiravo i 
        % dadi prima di richiamare la function
        
        % lavora sui due discendenti (offspring)
        % es ix = 3     nc1 ---ooxx  -->   mutnc1 --oooxx
        %               nc2 xxo-xxx  -->   mutnc2 xx--xxx

        % Attenzione! è diverso dal caso visto a lezione dove tiravo il
        % dado per ogni gene: qua tiro il dado per tutto il cromosoma
        % LO FA PER DUE CROMOSOMI PERCHE' LO STA FACENDO PER DUE CROMOSOMI
        % PER LA COPPIA 
        for i=1:size(chrom,2)

            % tiriamo il dado per vedere se mutare oppure no
            if rand(1) <= pmut
                ix1 = ceil(rand(1)*size(chrom,1)); % ESTRAGGO UN INDICE A CASO DELLA LUNGHEZZA DEL CROM
                ix2 = ceil(rand(1)*size(chrom,1)); % ESTRAGGO UN ALTRO INDICE A CASO DELLA LUNGHEZZA DEL CROM
                
                % faccio come la bubble sort ma su un solo elemento da
                % scambiare tra i due cromosomi
                tmp = chrom(ix1);           % variabile temporanea
                chrom(ix1) = chrom(ix2);
                chrom(ix2) = tmp;
            end
        end 
        newchrom = chrom;
    
    end 


end 

