addpath ../src

% circle
ndeNum    = 500;
angVec    = linspace(0, 2 * pi, ndeNum + 1);
iniNdeMat = [cos(angVec(1 : (end - 1))); ...
             sin(angVec(1 : (end - 1)))];
elmVtxMat = [1 : ndeNum; ...
             2 : ndeNum, 1];

% square, varifold representation
tgtNdeNum  = 1000;  % a multiple of 4
halfLen    = 0.8;
sideNdeNum = tgtNdeNum / 4;
sideVec    = linspace(-halfLen, halfLen, sideNdeNum + 1);
tgtNdeMat  = [sideVec(1 : sideNdeNum), halfLen * ones(1, sideNdeNum), ...
              sideVec(end : -1 : 2), -halfLen * ones(1, sideNdeNum); ...
              -halfLen * ones(1, sideNdeNum), sideVec(1 : sideNdeNum), ...
              halfLen * ones(1, sideNdeNum), sideVec(end : -1 : 2)];

tgtCenPosMat = 0.5 * (tgtNdeMat + circshift(tgtNdeMat, -1, 2));
tgtDirMat    = circshift(tgtNdeMat, -1, 2) - tgtNdeMat;
tgtElmVolVec = sqrt(sum(tgtDirMat.^2));
tgtUniDirMat = tgtDirMat ./ tgtElmVolVec;

% varifold pseudo-metric parameters
disObj.dfmElmVtxMat = elmVtxMat;
disObj.tgtCenPosMat = tgtCenPosMat;
disObj.tgtElmVolVec = tgtElmVolVec;
disObj.tgtUniDirMat = tgtUniDirMat;
disObj.cenKnlType   = 'C';
disObj.cenKnlWidth  = 0.1;
disObj.dirKnlType   = 'B';
disObj.dirKnlWidth  = 0;
disObj.disWgt       = 100;

% LDDMM parameters
knlOrder = 3;
knlWidth = 0.025;
timeEnd  = 1;
timeStp  = 0.05;

% optimization parameters
optObj.MaxIterations       = 100000;
optObj.OptimalityTolerance = 1e-2;
optObj.Wolfe1              = 0;
optObj.Wolfe2              = 0.5;
optObj.Columns             = 10000;
optObj.Verbose             = true;


% ----------- %
%  run LDDMM  %
% ----------- %

[ndeStk, vlcStk] = LDDMM(iniNdeMat, [], disObj, ...
                         knlOrder, knlWidth, timeEnd, timeStp, optObj);

rmpath ../src
