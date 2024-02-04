%--------------------------------------------------------------------------
% This is the function to compute the aggregated ranking along with the weights 
% for each of the MCDM ranking methods in the pool. It addition, it
% computes the consensus index and trust level for the final aggregated
% ranking.
% Input:
% R = includess the rankings produced by different MCDM methods (or any
% other rankings), each lies on a column of the matrix R.
% Outputs:
% R_star = the estimated R_star computed by the method
% finalRanking = the final ranking of alternatives 
% ConsensusIndex = an index for the consensus of different methods on the
%   final aggregated rankings 
% TrustLevel = an indicator of how reliable the final aggregated ranking is
% sigma = the width of the Welsch M-estimator 
%--------------------------------------------------------------------------
% Copyright @ Majid Mohammadi, 2020
% For more information, see the corresponding paper:
%   Mohammadi and Rezaei (2020), Ensemble Ranking: Aggregation of rankings produced 
%     by different multi-criteria decision-making methods, Omega, 2020.
%--------------------------------------------------------------------------

function [R_star,finalRanking,lambda, consensusIndex,trustLevel,sigma] = EnsembleRanking(R)

[alt_no,method_no] = size(R);
tol = 10e-10; %tolerance based on which the main loop is terminated
iter = 1000; % maximum number of iteration
i=0;
R_star = zeros(alt_no,1);

alpha = zeros(1,method_no); %half-quadratic auxiliary variable
lambda = alpha;%Lambda plays the role of weights in Ensemble Ranking
%% Main loop
while i < iter
    Error = R-repmat(R_star,1,method_no);
    Error_norm = sqrt(sum(Error.^2));
    if(sum(Error_norm) == 0)
        break;
    end
    sigma = (norm(Error_norm,2)^2 / (2*length(Error_norm)^2));
    
    alpha_old = alpha;
    alpha = delta(Error,sigma);
    
    R_star_old = R_star;
    lambda = alpha / sum(alpha);
    R_star = R*lambda';
    
    %Convergence conditions
    if norm(alpha(:)-alpha_old(:),2) < tol && norm(R_star-R_star_old) < tol
        break;
    end
    
    i = i + 1;
end

%%Computing confidence index and trust level
confMat = (normpdf(R - repmat(R_star,1,size(R,2)),0,sigma) / normpdf(0,0,sigma));

consensusIndex = sum(confMat(:)) / (method_no*alt_no);
trustMat = confMat * diag(lambda);
trustLevel = sum(trustMat(:)) / alt_no;

%% Computing final rankings
[~,ind] = sort(R_star,'ascend');
finalRanking = zeros(size(ind));
for i=1:length(ind)
    finalRanking(ind(i)) = i;
end

end

% the minizer function \delta() of the Welsch M-estimator
function E = delta(v,sigma)
[~,rnkrs] = size(v);
E = zeros(1,rnkrs);
for i=1:rnkrs
    E(i) = exp(-(norm(v(:,i)))^2 / (2*sigma^2));
end
end
