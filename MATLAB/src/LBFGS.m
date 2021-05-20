function [posNow, objNow, grdNow] = LBFGS(objgrdHdl, posNow, options)
%
% Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
% Date  : 06/10/2020

posNow = posNow(:);
varNum = length(posNow);

if nargin == 2, options.Verbose = false; end
options = setOptions(options, varNum);

wolfe1 = options.Wolfe1;
wolfe2 = options.Wolfe2;
itrMax = options.MaxIterations;
tolVal = options.OptimalityTolerance;
memNum = options.Columns;

dspMat = zeros(varNum, memNum);
dgdMat = zeros(varNum, memNum);

[objNow, grdNow] = objgrdHdl(posNow);
grdLen = norm(grdNow);

if options.Verbose
	fprintf('%5s   %13s  %13s  %13s  %9s\n', 'iter', 'f', '|grad f|', 'step length', 'fcn eval');
	fprintf('%s\n', repmat('-', 1, 62));
	fprintf('%5d:  %13.6e  %13.6e\n', 0, objNow, grdLen);
end

newIdx = 0;
for itrIdx = 1 : itrMax

	if grdLen < tolVal
		break;
	end

	if newIdx == 0
		HIniVal = 1;
	else
		HIniVal =  (dspMat(:, newIdx)' * dgdMat(:, newIdx)) ...
		         / (dgdMat(:, newIdx)' * dgdMat(:, newIdx));
	end

	if itrIdx <= memNum
		dirNow = getDirection(HIniVal, grdNow, dspMat, dgdMat, newIdx, itrIdx - 1, memNum);
	else
		dirNow = getDirection(HIniVal, grdNow, dspMat, dgdMat, newIdx, memNum, memNum);
	end

	[objNxt, posNxt, grdNxt, stpLen, objCnt, sucFlg] = ...
	   lineSearch(objgrdHdl, objNow, posNow, grdNow, dirNow, wolfe1, wolfe2, tolVal);

	if ~sucFlg
		return;
	end

	dspVec = posNxt - posNow;
	dgdVec = grdNxt - grdNow;

	if newIdx == memNum
		newIdx = 1;
	else
		newIdx = newIdx + 1;
	end

	dspMat(:, newIdx) = dspVec;
	dgdMat(:, newIdx) = dgdVec;

	objNow = objNxt;
	posNow = posNxt;
	grdNow = grdNxt;

	grdLen = norm(grdNow);

	if options.Verbose
		fprintf('%5d:  %13.6e  %13.6e  %13.6e  %9d\n', itrIdx, objNow, grdLen, stpLen, objCnt);
	end

end


% ---------------------------------------------------------------------------- %
%                                 Subfunctions                                 %
% ---------------------------------------------------------------------------- %

function options = setOptions(options, varNum)

if ~isfield(options, 'Wolfe1')
	options.Wolfe1 = 1e-4;
end

if ~isfield(options, 'Wolfe2')
	options.Wolfe2 = 0.9;
end

if ~isfield(options, 'MaxIterations')
	options.MaxIterations = 1000;
end

if ~isfield(options, 'OptimalityTolerance')
	options.OptimalityTolerance = 1e-6;
end

if ~isfield(options, 'Columns')
	options.Columns = min(100, varNum);
end

if ~isfield(options, 'Verbose')
	options.Verbose = false;
end
