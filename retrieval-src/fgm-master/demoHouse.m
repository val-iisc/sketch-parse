% A demo comparison of different graph matching methods on the CMU House dataset.
%
% Remark
%   The edge is directed and the edge feature is asymmetric. 
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-20-2012
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-08-2013

clear variables;
prSet(1);

%% src parameter
tag = 'house';
pFs = [1 100]; % frame index
nIn = [30 30] - 2; % randomly remove 2 nodes
parKnl = st('alg', 'cmum'); % type of affinity: only edge distance

%% algorithm parameter
[pars, algs] = gmPar(2);

%% src
wsSrc = cmumAsgSrc(tag, pFs, nIn, 'svL', 1);
asgT = wsSrc.asgT;

%% feature
parG = st('link', 'del'); % Delaunay triangulation for computing the graphs
parF = st('smp', 'n', 'nBinT', 4, 'nBinR', 3); % not used, ignore it
wsFeat = cmumAsgFeat(wsSrc, parG, parF, 'svL', 1);
[gphs, XPs, Fs] = stFld(wsFeat, 'gphs', 'XPs', 'Fs');

%% affinity
[KP, KQ] = conKnlGphPQU(gphs, parKnl);
K = conKnlGphKU(KP, KQ, gphs);
Ct = ones(size(KP));

%% undirected graph -> directed graph (for FGM-D)
gphDs = gphU2Ds(gphs);
KQD = [KQ, KQ; KQ, KQ];

%% GA
asgGa = gm(K, Ct, asgT, pars{1}{:});

%% PM
asgPm = pm(K, KQ, gphs, asgT);

%% SM
asgSm = gm(K, Ct, asgT, pars{3}{:});

%% SMAC
asgSmac = gm(K, Ct, asgT, pars{4}{:});

%% IPFP-U
asgIpfpU = gm(K, Ct, asgT, pars{5}{:});

%% IPFP-S
asgIpfpS = gm(K, Ct, asgT, pars{6}{:});

%% RRWM
asgRrwm = gm(K, Ct, asgT, pars{7}{:});

%% FGM-U
asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});

%% FGM-D
asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});

%% print accuracy and objective
fprintf('GA    : acc %.2f, obj %.2f\n', asgGa.acc, asgGa.obj);
fprintf('PM    : acc %.2f, obj %.2f\n', asgPm.acc, asgPm.obj);
fprintf('SM    : acc %.2f, obj %.2f\n', asgSm.acc, asgSm.obj);
fprintf('SMAC  : acc %.2f, obj %.2f\n', asgSmac.acc, asgSmac.obj);
fprintf('IPFP-U: acc %.2f, obj %.2f\n', asgIpfpU.acc, asgIpfpU.obj);
fprintf('IPFP-S: acc %.2f, obj %.2f\n', asgIpfpS.acc, asgIpfpS.obj);
fprintf('RRWM  : acc %.2f, obj %.2f\n', asgRrwm.acc, asgRrwm.obj);
fprintf('FGM-U : acc %.2f, obj %.2f\n', asgFgmU.acc, asgFgmU.obj);
fprintf('FGM-D : acc %.2f, obj %.2f\n', asgFgmD.acc, asgFgmD.obj);

%% show correspondence result
rows = 1; cols = 1;
Ax = iniAx(1, rows, cols, [400 * rows, 900 * cols], 'hGap', .1, 'wGap', .1);
parCor = st('cor', 'ln', 'mkSiz', 7, 'cls', {'y', 'b', 'g'});
shAsgImg(Fs, gphs, asgFgmD, asgT, parCor, 'ax', Ax{1}, 'ord', 'n');
title('result of FGM-D');

%% show affinity
rows = 1; cols = 3;
Ax = iniAx(2, rows, cols, [200 * rows, 200 * cols]);
shAsgK(K, KP, KQ, Ax);

%% show correpsondence matrix
asgs = {asgT, asgGa, asgPm, asgSm, asgSmac, asgIpfpU, asgIpfpS, asgRrwm, asgFgmU, asgFgmD};
rows = 2; cols = 5;
Ax = iniAx(3, rows, cols, [250 * rows, 250 * cols]);
shAsgX(asgs, Ax, ['Truth', algs, 'FGM-A']);
