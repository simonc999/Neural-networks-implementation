function out=tournamentk(fitvect,pk)
%out=tournament(fitvect,pk)
%GA module: Tournament selection of ONE chromosome given the fitness
% vector of the population. Returns an index of a chromosome in the
% population
%pk represents the threshold value for selecting the best chromosome in
%case a random number r is < k, the worse otherwise.
%SR 2004

if nargin < 2
    pk=0.7;
end

draw=ceil(length(fitvect)*rand(2,1)); %Select two random numbers (chromosome indices)
if rand(1)<pk
    [m,i]=max(fitvect(draw)); %Select the best chromosome
else
    [m,i]=min(fitvect(draw)); %Select the worse chromosome
end
    
out=draw(i); %Select index of the selected chromosome

