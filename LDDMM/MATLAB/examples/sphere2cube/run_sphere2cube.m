addpath ../src

% sphere
load sphereData

% cube, varifold representation
load cubeData

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
knlWidth = 0.25;
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
