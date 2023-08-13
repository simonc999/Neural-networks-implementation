function out=twop_cross(chr1,chr2) % two point cross
%function out=singlep_cross(chr1,chr2)
%SR 2005

if size(chr1,2)>1 %Transform in Column vectors
    chr1=chr1';
    chr2=chr2';
end

%Select random index and rescale for the number of genes in the chromosome;
idxs=sort(ceil((size(chr1,1)-1)*rand(1,2))); %There are one less crossover points than elements!
% -1 per togliere l'ultimo indice e ceil per evitare lo zero
idx1=idxs(1);
idx2=idxs(2);

%Switch the first parts of the chromosomes
out2=[chr1(1:idx1); chr2(idx1+1:idx2); chr1(idx2+1:end)]; %Cross from index+1 to end of chromosome
out1=[chr2(1:idx1); chr1(idx1+1:idx2); chr2(idx2+1:end)];
%Return the two offsprings
out=[out1 out2];
