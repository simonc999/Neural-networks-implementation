% Un GA lo posso usare anche per un problema di identificazione di un 
% sistema che possa replicare il comportamento di un sistema dinamico. 
% Considero una struttura predefinita di una funzione di trasferimento, 
% quindi decido a priori di avere una funzione di trasferimento abbastanza 
% generica costituita da 1 ritardo, 1 guadagno, 2 zeri, 3 poli (abbastanza 
% generica per potersi adattare a diverse situazioni). Ognuno di questi è 
% un parametro e quindi può essere un gene di un algoritmo genetico.
% Il cromosoma è costituito da G, tau, z1, z2, p1, p2, p3… e quindi 
% costruire una funzione di fitness che premi la minimizzazione di un 
% errore quadratico medio tra la simulazione del sistema e la simulazione 
% del sistema che usa come parametri i geni del cromosoma.
clc
clear
close all



% impostazione del seed
rng(123456)

% ------------------------------------------------------------------- %
% costruzione del training set: funzione di trasferimento nota
G_id = 2;
z1_id = 1;
z2_id =4;
p1_id = 5;
p2_id = 0.2;
p3_id = 1;
tau_id = 3;
syms s
[c] = double(fliplr(coeffs((s+p1_id)*(s+p2_id)*(s+p3_id))));


num_id = G_id*conv([z1_id 1],[z2_id 1]);                % zero (termine di grado 0 e termine di grado 1)
den_id = c;  % 'prodotto' dei poli 
sys_id = tf(num_id,den_id,'InputDelay',tau_id);          % tf: transfr function
% bode(sys_id)

Tc = 0.05;                       % discretizzazione
t = 0:Tc:30;                     % vettore dei tempi (sono 601=(30/0.05)+1 sec)

u0 = [zeros(1,20) ones(1,100) zeros(1,200) ones(1,80) zeros(1,80) 0:Tc:1];   
u1 = [sin(2*pi*t(1:100))];
u = [u0 u1];                      % ingresso

y = lsim(sys_id,u,t);                % risposta del sistema

% figure('Name','Training set');
% plot(t,u,t,y)
% legend('Ingresso','Ripsosta')


% ------------------------------------------------------------------- %
% PARAMETRI INIZIALI

% numero di generazioni
% ngen = 10000;
% ngen = 10;
 ngen = 100;
%ngen = 300;

% decidiamo di usare una funzione di strasferimento genereica con 
% chrom = [G tau z1 p1 p2 p3];
npar = 7;
popsize = 100;
pcross = 0.7;
pmut = 0.05; 


% --------------------------------------------------------------------- %
% POPOLAZIONE INIZIALE
% costruzione della popolazione iniziale
pop = tsf_pop(popsize,npar);

% --------------------------------------------------------------------- %
% FITNESS
fitvec = tsf_fit(pop,y,u,t); 


% --------------------------------------------------------------------- %
% ELITISMO
% sostanzialmente devo mettere in ordine il vettore di fitness
[~, sortix] = sort(fitvec);


% --------------------------------------------------------------------- %
% EVOLUZIONE
% iniziamo con l'evoluzione attraverso le generazioni
for i=1:ngen

    
    % tengo traccia di cosa sta succedendo
    maxfit = max(fitvec);       % idealmente non voglio che diminuisca
    meanfit = mean(fitvec);
    
    % -------- %
    

    % costruisco la nuova popolazione lavorando per coppie
    % metto 25 coppie -1 coppia data già per elitismo
    for j=1:((popsize/2)-1)  % -1 to adjust for elitism
        
        % -------------------------------------------------------- %
        % TOURNAMENT
        
        CR = pop(:,sortix(end-1:end));
        chr1 = CR(:,1);
        chr2 = CR(:,2);          
               
        newchrom =  twop_cross(chr1,chr2);        
        WORST = pop(:,sortix(1:2));

        pop(:,sortix(1:2))=[];
        
        % !!! prima gli offspring poi i peggiori
        cromosomi = [newchrom WORST];

        fitness2evaluate = tsf_fit(cromosomi,y,u,t);
        
        % !!! prima gli offspring poi i peggiori
        [B,I] = maxk(fitness2evaluate,2);

        pop = [pop cromosomi(:,I)];
        


        

    end 
    
    % ---------------------------------------------------------------- %
    % RICALCOLO FITNESS
    % siamo sempre nella i-esima popolazione, devo ricalcolare la nuova
    % fitness
   
    fitvec = tsf_fit(pop,y,u,t);
    
    % ---------------------------------------------------------------- %
    % ELITISMO
    [sortfit, sortix] = sort(fitvec);
end 



% ---------------------------------------------------------- %
% grafico 
% mostriamo la mappa percorsa secondo il miglior cromosoma
best = pop(:,sortix(end)); % miglior cromosoma
out = tsf_out(best,t,u);

figure('Name','Best Chromosome')
plot(t,u,t,y)
hold on 
plot(t,out)
legend('ingresso','risp ideale','risp calcolata')


% ---------------------------------------------------------- %
% visualizzazione dei parametri
ideale = [G_id z1_id z2_id p1_id p2_id p3_id tau_id];
clc
fprintf('parametri ideali %d %d %d %d %d %f %d \n', ideale)
fprintf('parametri trovati %2f  %2f %2f %2f %2f %2f %2f \n', best)


% --------------------------------------------------------------------- %
% funzione creazione popolazione
% --------------------------------------------------------------------- %
function pop = tsf_pop(popsize,npar)

        % creo la popolazione cromosomica come una permutazione di indici
        % di parametri della funzione di trasferimento
        for i=1:popsize
            pop(:,i) = randperm(10,npar)*rand(1)';     
            % numeri casuali da 1 a 10 
            % e ne voglio estrarre 5                                                       
        end 
 end 


% --------------------------------------------------------------------- %
% funzione calcolo vettore di fitness
% --------------------------------------------------------------------- %
    function fitvec = tsf_fit(pop,y,u,t)
        
        % la fitness viene calcolata come 1 fratto la distanza complessiva 
        % del percorso
        for i =1:size(pop,2)
           
            chrom = pop(:,i);   % cromosoma i-esimo

            % estrazione dei parametri
            G = chrom(1);
            tau = chrom(2);
            z1 = chrom(3);
            z2 = chrom(4);
            p1 = chrom(5);
            p2 = chrom(6);
            p3 = chrom(7);

            syms s
            c = double(fliplr(coeffs((s+p1)*(s+p2)*(s+p3))));
            % creazione della funzione di trasferimento
            num = G*conv([z1 1],[z2 1]);               
            den = c;
           
            sys = tf(num,den,'InputDelay',tau); 
            
            % ingresso
            % in ingresso le do lo stesso vettore cìche gli do sopra

            % fenotipo
            yCh = lsim(sys,u,t); 

            % fitness
            % avrà fitness > quello con errore quadratico medio piu piccolo
            eqm = sum(((yCh-y).^2)/size(y,1));  % errore quadratico medio
            fitvec(i) = 1/(eqm);
            % fitvec(i) = 1/sum(abs(yCh'-y));
        end 

    end 

% --------------------------------------------------------------------- %
% funzione Turnament K
% --------------------------------------------------------------------- %

    function out=tournamentk(fitvect,pk)

    % out=tournament(fitvect,pk)
    % GA module: Tournament selection of ONE chromosome given the fitness
    % vector of the population. Returns an index of a chromosome in the
    % population
    
    % pk represents the threshold value for selecting the best chromosome in
    % case a random number r is < k, the worse otherwise.
    
        if nargin < 2
            pk = 0.7;   % deve essere compreso tra [0 1] e > 0.5
        end
        
        draw = ceil(length(fitvect)*rand(2,1)); % Select two random numbers (chromosome indices)
        
        if rand(1)<pk
            [m,i] = max(fitvect(draw)); %Select the best chromosome
        else
            [m,i] = min(fitvect(draw)); %Select the worse chromosome
        end
            
    out = draw(i); % Select index of the selected chromosome

    end 


% --------------------------------------------------------------------- %
% funzione twop_cross
% --------------------------------------------------------------------- %
    function out = twop_cross(chr1,chr2)
        
        % controllo sulle dimensioni: se è un vettore riga lo traspone a
        % colonna, altrimenti lo lasci a vettore colonna
        if size(chr1,2)>1 %Transform in Column vectors
            chr1=chr1';
            chr2=chr2';
        end
        
        % Select random index and rescale for the number of genes in the chromosome;
        % usa ceil per evitare lo zero
        % usa size-1 per evitare l'ultimo elemento: There are one less crossover points than elements!
        idxs = sort(ceil((size(chr1,1)-1)*rand(1,2)));
        idx1 = idxs(1);
        idx2 = idxs(2);
        
        % Switch the first parts of the chromosomes
        % attenzione che io cambierei i nomi
        out2 = [chr1(1:idx1); chr2(idx1+1:idx2); chr1(idx2+1:end)]; %Cross from index+1 to end of chromosome
        out1 = [chr2(1:idx1); chr1(idx1+1:idx2); chr2(idx2+1:end)];
    
        % Return the two offsprings
        out=[out1 out2];
    
    end 


% --------------------------------------------------------------------- %
% funzione mutazione
% --------------------------------------------------------------------- %
    function newchrom = tsf_mut(chrom, pmut)

        % Attenzione! è diverso dal caso del cross over in cui tiravo i 
        % dadi prima di richiamare la function
        
        % lavora sui due discendenti (offspring)
        % es ix = 3     nc1 ---ooxx  -->   mutnc1 --oooxx
        %               nc2 xxo-xxx  -->   mutnc2 xx--xxx

        % Attenzione! è diverso dal caso visto a lezione dove tiravo il
        % dado per ogni gene: qua tiro il dado per tutto il cromosoma, 
        % size(chrom,2) = 2 quindi lo fa per i due cromosomi della coppia
        for i=1:size(chrom,2)

            % tiriamo il dado per vedere se mutare oppure no
            if rand(1) <= pmut
                ix1 = ceil(rand(1)*size(chrom,1));
                ix2 = ceil(rand(1)*size(chrom,1));
                
                % faccio come la bubble sort ma su un solo elemento da
                % scambiare tra i due cromosomi
                tmp = chrom(ix1);           % variabile temporanea
                chrom(ix1) = chrom(ix2);
                chrom(ix2) = tmp;
            end
        end 
        newchrom = chrom;
    
    end 

% --------------------------------------------------------------------- %
% funzione uscita calcolatore
% --------------------------------------------------------------------- %
    function out = tsf_out(best,t,u)

            % estrazione dei parametri
            Gb = best(1);
            taub = best(2);
            z1b = best(3);
            z2b = best(4);
            p1b = best(5);
            p2b = best(6);
            p3b = best(7);
            syms s
            c = double(fliplr(coeffs((s+p1b)*(s+p2b)*(s+p3b))));
            % creazione della funzione di trasferimento
            numb = Gb*conv([z1b 1],[z2b 1]);               
            denb = c;
            sysb = tf(numb,denb,'InputDelay',taub); 
            
            % fenotipo
            out = lsim(sysb,u,t); 
            
    end 

%end