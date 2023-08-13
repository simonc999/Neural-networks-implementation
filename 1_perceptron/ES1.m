%% ESERCITAZIONE 1 

% vogliamo implementare in matlab algoritmo del percettrone, vogliamo
% toccare un problema di classificazione linearmente separabile e chiedere
% al percettrone di risolverlo. Vogliamo quindi definire una retta y= a x +
% b e questa retta suddivide il piano in due regioni, supponiamo che sopra
% la retta ci sia C1 mentre sotto C2 e quindi scriviamo funzione in matlab
% che generi una serie di punti casuali in una regione limitata nel piano e
% che poi li classifichi con la retta. Dopodichè alleneremo la rete con i
% punti trovati e le uscite desiderate corrispondenti. Quando l'algoritmo
% si arresta fermo il grafico oltre che del dataset anche di quella
% individuata dal percettrone. 

clear
close all
clc

eta=0.2;
coeff = [2 1];
% numero dei punti
N=20;

% genero punti in [-2 2][-2 2]
% i punti rappresentati saranno costituiti da 2 attributi e x1 su asse 
% orizzontale x e x2 su asse verticale y
x1= 4*rand(N,1)-2;
x2= 4*rand(N,1)-2;


x=-2:0.05:2;
y=coeff(1).*x + coeff(2);
% classificazione corretta
% x2- coeff(1)*x1 - coeff(2)>=0 sono in classe 1
% x2- coeff(1)*x1 - coeff(2)<0 sono in classe 2

% converto i logical in double, se è maggiore o uguale a 0 allora sono in
% classe 1
d = double (x2 - coeff(1)*x1 - coeff(2)>=0);

figure
plot(x1,x2,'xk')
hold on 
plot(x,y,'r')
hold on 

% uso il vettore delle uscite desiderate come come soglia
% se è vera il punto viene graficato altrimenti no 
plot(x1(d==1),x2(d==1),'og') % classe 1
hold on 
plot(x1(d==0),x2(d==0),'ob') % classe 0

% percettrone
% inizializzo wT

w=[0 0 0];  % w(1) rappresenta il bias

xtrain = [ones(N,1) x1 x2]'; % ogni colonna è un esempio


finito=0; %flag arresto
epoca = 0;

% l'algoritmo si interrompe quando conduco un'intera epoca senza errore
% tilde vuol dire not 
while ~finito
    epoca = epoca+1;
    for i =1:N
        % y funzione di attivazione applicata al nostro training
        y=double(w*xtrain(:,i)>=0);
        e(i)=d(i)-y;
        % applichiamo la regola Delta
        dw=eta*e(i)*xtrain(:,i);
        w=w+dw';
    end
    
    if sum(abs(e))==0
        finito=1;
    end
end

% verifico retta attesa
hold on
yp=-w(2)/w(3)*x-w(1)/w(3);
plot(x,yp,'c')

%%sprintf('la retta individuata è y=%2.2f*x+%2.2f dopo %d epoche',-w(2)/w(3)*x-w(1)/w(3))
