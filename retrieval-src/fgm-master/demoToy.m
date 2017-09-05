% A demo comparison of different graph matching methods on the synthetic dataset.
%
% Remark
%   The edge is directed and the edge feature is asymmetric.
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-20-2012
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-07-2013

clear variables;
prSet(1);

%% src parameter
tag = 1; 
nIn = 10; % #inliers
nOuts = [0 0]; % #outliers
egDen = .5; % edge density
egDef = 0; % edge deformation
parKnl = st('alg', 'toy'); % type of affinity: synthetic data

%% algorithm parameter
[pars, algs] = gmPar(2);

%% src
wsSrc = toyAsgSrcD(tag, nIn, nOuts, egDen, egDef);
[gphs, asgT] = stFld(wsSrc, 'gphs', 'asgT');

%% affinity
[KP, KQ] = conKnlGphPQD(gphs, parKnl); % node and edge affinity
K = conKnlGphKD(KP, KQ, gphs); % global affinity
Ct = ones(size(KP)); % mapping constraint (default to a constant matrix of one)

%% directed graph -> undirected graph (for fgmU and PM)
gphUs = gphD2Us(gphs);
[~, KQU] = knlGphKD2U(KP, KQ, gphUs);

%% Truth
asgT.obj = asgT.X(:)' * K * asgT.X(:);
asgT.acc = 1;

%% GA
asgGa = gm(K, Ct, asgT, pars{1}{:});

%% PM
asgPm = pm(K, KQU, gphUs, asgT);

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
asgFgmU = fgmU(KP, KQU, Ct, gphUs, asgT, pars{8}{:});
X = asgFgmU.X;
asgFgmU.obj = X(:)' * K * X(:);

%% FGM-D
asgFgmD = fgmD(KP, KQ, Ct, gphs, asgT, pars{9}{:});

%% print accuracy and objective
fprintf('Truth : acc %.2f, obj %.2f\n', asgT.acc, asgT.obj);
fprintf('GA    : acc %.2f, obj %.2f\n', asgGa.acc, asgGa.obj);
fprintf('PM    : acc %.2f, obj %.2f\n', asgPm.acc, asgPm.obj);
fprintf('SM    : acc %.2f, obj %.2f\n', asgSm.acc, asgSm.obj);
fprintf('SMAC  : acc %.2f, obj %.2f\n', asgSmac.acc, asgSmac.obj);
fprintf('IPFP-U: acc %.2f, obj %.2f\n', asgIpfpU.acc, asgIpfpU.obj);
fprintf('IPFP-S: acc %.2f, obj %.2f\n', asgIpfpS.acc, asgIpfpS.obj);
fprintf('RRWM  : acc %.2f, obj %.2f\n', asgRrwm.acc, asgRrwm.obj);
fprintf('FGM-U : acc %.2f, obj %.2f\n', asgFgmU.acc, asgFgmU.obj);
fprintf('FGM-D : acc %.2f, obj %.2f\n', asgFgmD.acc, asgFgmD.obj);

%% show correspondence matrix
asgs = {asgT, asgGa, asgPm, asgSm, asgSmac, asgIpfpU, asgIpfpS, asgRrwm, asgFgmU, asgFgmD};
rows = 2; cols = 5;
Ax = iniAx(1, rows, cols, [250 * rows, 250 * cols]);
shAsgX(asgs, Ax, ['Truth' algs]);
