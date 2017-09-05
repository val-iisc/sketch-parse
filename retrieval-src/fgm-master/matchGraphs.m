%% Function to evaluate similarity between 2 graphs

function score = matchGraphs(Kmat, Ct)
    load('pars.mat');
    asgRrwm = gm(Kmat, Ct, [], pars{7}{:});
    
    indx = asgRrwm.X';
    indx = indx(:);
    diagK = diag(Kmat);
    
    NStotal = sum(diagK(find(indx)));   % total node score
   
    NSlocal = sum(diagK(find(indx(1:end-1))));   % local node score
    NSglobal = NStotal-NSlocal;   % global node score

%    indices = find(indx);
%    NStotal = sum(diagK(indices));   % total node score
%    NSlocal = sum(diagK(indices(1:end-2)));   % local node score
%    NSaction = sum(diagK(indices(end-2)));   % action node score
%    NSglobal = NStotal-NSlocal-NSaction;   % global node score
   %score =1 
    k = 0.9
	l = 0.1
%	score = k*NSlocal + l* NSglobal +(1-k-l)*(asgRrwm.obj-NStotal)
	score = NSlocal+NSglobal+asgRrwm.obj-NStotal
	%score = NSlocal
end
