function [finalRank] = hpa_func(sim,topK)
galleryNum = size(sim, 1);
rankerNum = size(sim, 2);
averageRank = sum(sim, 2);
averageRank = averageRank ./ max(averageRank);
[~,pseudoRanklist] = sort(averageRank);
[~,pseudoRank] = sort(pseudoRanklist);
[~,ranklist] = sort(-sim);
NDCG = zeros(rankerNum,1);
for i=1:rankerNum
    for j=1:topK
        NDCG(i) = NDCG(i) + averageRank(ranklist(j,i)) * log(2) / log(i+1);
    end
end

[~,NDCGrank] = sort(-NDCG,1);

finalRank = zeros(galleryNum,1);
% for i=1:5
for i=1:rankerNum
    finalRank = NDCG(NDCGrank(i)) * sim(:,NDCGrank(i)) + finalRank;
end
[~,finalRank] = sort(-finalRank);
[~,finalRank] = sort(finalRank);