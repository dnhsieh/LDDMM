function [objTry, posTry, grdTry, stpTry, objCnt, sucFlg] = ...
   lineSearch(objgrdHdl, objNow, posNow, grdNow, dirNow, wolfe1, wolfe2, tolVal)
%LINESEARCH was modified from HANSO 2.0
%
% Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
% Date  : 05/20/2020

relTry = 1.0;
relLft = 0.0;
relRgt = inf;

sucFlg = true;
slpNow = grdNow' * dirNow;
if ( slpNow > -eps )

	warning('Not a descent direction. Quit LBFGS.');
	
	objTry = objNow;
	posTry = posNow;
	grdTry = grdNow;
	stpTry = inf;
	objCnt = 0;
	sucFlg = false;

	return;

end

dirLen = norm(dirNow);

% arbitrary heuristic limits
bisMax = max(50, log2(1e5 * dirLen));
epdMax = max(10, log2(1e5 / dirLen));

bisCnt = 0;
epdCnt = 0;
objCnt = 0;

while bisCnt <= bisMax && epdCnt <= epdMax

	posTry = posNow + relTry * dirNow;
	[objTry, grdTry] = objgrdHdl(posTry);
	objCnt = objCnt + 1;

	if norm(grdTry) < tolVal
		stpTry = relTry * dirLen;
		return;
	end

	slpTry = grdTry' * dirNow;
	if objTry >= objNow + wolfe1 * relTry * slpNow
		relRgt = relTry;
	elseif slpTry <= wolfe2 * slpNow
		relLft = relTry;
	else
		stpTry = relTry * dirLen;
		return;
	end

	if isinf(relRgt)
		relTry = 2 * relTry;
		epdCnt = epdCnt + 1;
	else
		relTry = 0.5 * (relLft + relRgt);
		bisCnt = bisCnt + 1;
	end

end

if isinf(relRgt)
	warning(['Line search failed to bracket a point satisfying weak Wolfe conditions. ' ...
	         'Function may be unbounded below. Quit LBFGS.']);
	stpTry = relTry * dirLen;
	sucFlg = false;
else
	warning(['Line search failed to satisfy weak Wolfe conditions, although ' ...
	         'a point satisfying conditions was bracketed in [%e, %e]. Quit LBFGS.'], relLft, relRgt)
	stpTry = relTry * dirLen;
	sucFlg = false;
end

