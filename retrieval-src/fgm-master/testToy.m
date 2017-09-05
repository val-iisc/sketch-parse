% Testing the performance of different graph matching methods on the synthetic dataset.
% This is a similar function used for reporting (Fig. 4) the first experiment (Sec 5.1) in the CVPR 2012 paper.
%
% History    
%   create   -  Feng Zhou (zhfe99@gmail.com), 09-01-2012
%   modify   -  Feng Zhou (zhfe99@gmail.com), 05-08-2013

clear variables;
prSet(4);

%% save flag
svL = 2; % change svL = 1 if you want to re-run the experiments.

%% algorithm parameter
tagAlg = 2;
[~, algs] = gmPar(tagAlg);

%% run 1 (outliers)
tagSrc = 1;
[~, val1s] = toyAsgPar(tagSrc);
wsRun1 = toyAsgRunD(tagSrc, tagAlg, 'svL', svL);
[Me1, Dev1, ObjMe1, ObjDev1] = stFld(wsRun1, 'Me', 'Dev', 'ObjMe', 'ObjDev');

%% run 2 (deformation)
tagSrc = 2;
[~, val2s] = toyAsgPar(tagSrc);
wsRun2 = toyAsgRunD(tagSrc, tagAlg, 'svL', svL);
[Me2, Dev2, ObjMe2, ObjDev2] = stFld(wsRun2, 'Me', 'Dev', 'ObjMe', 'ObjDev');

%% run 3 (sparseness)
tagSrc = 3;
[~, val3s] = toyAsgPar(tagSrc);
wsRun3 = toyAsgRunD(tagSrc, tagAlg, 'svL', svL);
[Me3, Dev3, ObjMe3, ObjDev3] = stFld(wsRun3, 'Me', 'Dev', 'ObjMe', 'ObjDev');

%% show accuracy & objective
rows = 1; cols = 7;
Ax = iniAx(1, rows, cols, [250 * rows, 250 * cols]);

shCur(Me1, Dev1, 'ax', Ax{1}, 'dev', 'y');
xticks = 1 : 2 : size(Me1, 2);
setAxTick('x', '%d', xticks, val1s(xticks));
set(gca, 'ylim', [0 1.1], 'ytick', 0 : .2 : 1, 'xlim', [.5, size(Me1, 2) + .5]);
axis square;
xlabel('outlier');
ylabel('accuracy');

shCur(ObjMe1, ObjDev1, 'ax', Ax{2}, 'dev', 'y');
xticks = 1 : 2 : size(ObjMe1, 2);
setAxTick('x', '%d', xticks, val1s(xticks));
set(gca, 'ylim', [.4 1.05], 'ytick', .2 : .1 : 1, 'xlim', [.5, size(Me1, 2) + .5]);
axis square;
xlabel('outlier');
ylabel('objective ratio');

shCur(Me2, Dev2, 'ax', Ax{3}, 'dev', 'y');
xticks = 1 : 2 : size(Me2, 2);
setAxTick('x', '%.2f', xticks, val2s(xticks));
set(gca, 'ylim', [0 1.1], 'ytick', 0 : .2 : 1, 'xlim', [.5, size(Me2, 2) + .5]);
axis square;
xlabel('edge deformation');
ylabel('accuracy');

shCur(ObjMe2, ObjDev2, 'ax', Ax{4}, 'dev', 'y');
xticks = 1 : 2 : size(ObjMe2, 2);
setAxTick('x', '%.2f', xticks, val2s(xticks));
set(gca, 'ylim', [.4 1.05], 'ytick', .2 : .1 : 1, 'xlim', [.5, size(Me2, 2) + .5]);
axis square;
xlabel('edge deformation');
ylabel('objective ratio');

shCur(Me3, Dev3, 'ax', Ax{5}, 'dev', 'y');
xticks = 1 : 2 : size(Me3, 2);
setAxTick('x', '%.1f', xticks, val3s(xticks));
set(gca, 'ylim', [0 1.1], 'ytick', 0 : .2 : 1, 'xlim', [.5, size(Me3, 2) + .5]);
axis square;
xlabel('edge sparseness');
ylabel('accuracy');

shCur(ObjMe3, ObjDev3, 'ax', Ax{6}, 'dev', 'y');
xticks = 1 : 2 : size(ObjMe3, 2);
setAxTick('x', '%.1f', xticks, val3s(xticks));
set(gca, 'ylim', [.4 1.05], 'ytick', .2 : .1 : 1, 'xlim', [.5, size(Me3, 2) + .5]);
axis square;
xlabel('edge sparseness');
ylabel('objective ratio');

shCur(Me1, Dev1, 'ax', Ax{7}, 'dev', 'n', 'algs', algs);
set(Ax{7}, 'visible', 'off');
cla;
