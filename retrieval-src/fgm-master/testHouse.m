% Testing the performance of different graph matching methods on the CMU House dataset.
% This is the same function used for reporting (Fig. 4) the first experiment (Sec 5.1) in the CVPR 2013 paper.
%
% History
%   create   -  Feng Zhou (zhfe99@gmail.com), 09-01-2011
%   modify   -  Feng Zhou (zhfe99@gmail.com), 05-08-2013

clear variables;
prSet(4);

%% save flag
svL = 1; % change svL = 1 if you want to re-run the experiments.

%% algorithm parameter
tagAlg = 2;
[~, algs] = gmPar(tagAlg);

%% run 1 (perfect graphs, no noise)
tagSrc = 1;
[~, val1s] = cmumAsgPair(tagSrc);
wsRun1 = cmumAsgRun(tagSrc, tagAlg, 'svL', svL);
[Me1, Dev1, ObjMe1, ObjDev1] = stFld(wsRun1, 'Me', 'Dev', 'ObjMe', 'ObjDev');

%% run 2 (randomly remove nodes)
tagSrc = 2;
[~, val2s] = cmumAsgPair(tagSrc);
wsRun2 = cmumAsgRun(tagSrc, tagAlg, 'svL', svL);
[Me2, Dev2, ObjMe2, ObjDev2] = stFld(wsRun2, 'Me', 'Dev', 'ObjMe', 'ObjDev');

%% show accuracy & objective
rows = 1; cols = 5;
Ax = iniAx(1, rows, cols, [250 * rows, 250 * cols]);

shCur(Me1, Dev1, 'ax', Ax{1}, 'dev', 'y');
xticks = 1 : 3 : size(Me1, 2);
setAxTick('x', '%d', xticks, val1s(xticks));
set(gca, 'ylim', [.27 1.05], 'ytick', .2 : .2 : 1, 'xlim', [.5, size(Me1, 2) + .5]);
axis square;
xlabel('baseline');
ylabel('accuracy');

shCur(ObjMe1, ObjDev1, 'ax', Ax{2}, 'dev', 'y');
xticks = 1 : 3 : size(ObjMe1, 2);
setAxTick('x', '%d', xticks, val1s(xticks));
set(gca, 'ylim', [.3 1.05], 'ytick', .2 : .2 : 1, 'xlim', [.5, size(Me1, 2) + .5]);
axis square;
xlabel('baseline');
ylabel('objective');

shCur(Me2, Dev2, 'ax', Ax{3}, 'dev', 'y');
xticks = 1 : 3 : size(Me2, 2);
setAxTick('x', '%d', xticks, val1s(xticks));
set(gca, 'ylim', [.27 1.05], 'ytick', .2 : .2 : 1, 'xlim', [.5, size(Me1, 2) + .5]);
axis square;
xlabel('baseline');
ylabel('accuracy');

shCur(ObjMe2, ObjDev2, 'ax', Ax{4}, 'dev', 'y');
xticks = 1 : 3 : size(ObjMe1, 2);
setAxTick('x', '%d', xticks, val1s(xticks));
set(gca, 'ylim', [.3 1.05], 'ytick', .2 : .2 : 1, 'xlim', [.5, size(Me1, 2) + .5]);
axis square;
xlabel('baseline');
ylabel('objective');

shCur(Me1, Dev1, 'ax', Ax{5}, 'dev', 'n', 'algs', algs);
set(Ax{5}, 'visible', 'off');
cla;
