function elmIfoMat = getElmIfo(elmVtxMat, ndeNum)

if size(elmVtxMat, 1) == 2

	adjCntVec = histcounts(elmVtxMat(:), 1 : (ndeNum + 1));
	elmIfoMat = zeros(1 + max(adjCntVec) * 2, ndeNum, 'int32');
	
	for lmkIdx = 1 : ndeNum
	
		[lclIdxVec, elmIdxVec] = find(elmVtxMat == lmkIdx);
	
		adjNum = length(elmIdxVec);
		elmIfoMat(1, lmkIdx) = adjNum;
	
		for adjIdx = 1 : adjNum
			if lclIdxVec(adjIdx) == 1
				elmIfoMat(1 + 2 * (adjIdx - 1) + (1 : 2), lmkIdx) = [elmIdxVec(adjIdx) - 1, -1];
			else
				elmIfoMat(1 + 2 * (adjIdx - 1) + (1 : 2), lmkIdx) = [elmIdxVec(adjIdx) - 1,  1];
			end
		end
	
	end

elseif size(elmVtxMat, 1) == 3

	adjCntVec = histcounts(elmVtxMat(:), 1 : (ndeNum + 1));
	elmIfoMat = zeros(1 + max(adjCntVec) * 3, ndeNum, 'int32');
	
	for lmkIdx = 1 : ndeNum
	
		[lclIdxVec, elmIdxVec] = find(elmVtxMat == lmkIdx);
	
		adjNum = length(elmIdxVec);
		elmIfoMat(1, lmkIdx) = adjNum;
	
		for adjIdx = 1 : adjNum
	
			lclIdx    = lclIdxVec(adjIdx);
			elmIdx    = elmIdxVec(adjIdx);
			elmVtxVec = elmVtxMat(:, elmIdx);
	
			if lclIdx == 1
				elmIfoMat(1 + 3 * (adjIdx - 1) + (1 : 3), lmkIdx) = [elmIdx; elmVtxVec([3, 2])] - 1;
			elseif lclIdx == 2
				elmIfoMat(1 + 3 * (adjIdx - 1) + (1 : 3), lmkIdx) = [elmIdx; elmVtxVec([1, 3])] - 1;
			else
				elmIfoMat(1 + 3 * (adjIdx - 1) + (1 : 3), lmkIdx) = [elmIdx; elmVtxVec([2, 1])] - 1;
			end
	
		end
	
	end

else

	error('The number of rows of ELMVTXMAT must be 2 or 3.')

end
