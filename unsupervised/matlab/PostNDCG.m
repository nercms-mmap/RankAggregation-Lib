% for calculagraph
num_runs = 10;
% calculagraph
calculagraph = zeros(1, num_runs);

for times = 1:num_runs

    % 记录开始时间
    start_time = tic;

    dataset_name = 'ice cream';

    sim = importdata('D:\Code of RA\Preflib\results\ice-cream\ice-cream.mat');
    rankernum = size(sim,1);
    querynum = size(sim,2);
    item_num = size(sim,3);
    
    [~,ranklist] = sort(-sim,3);
    [~,rank] = sort(ranklist,3);

    
    result = zeros(querynum,item_num);
    ndcglist = zeros(rankernum,rankernum);
    ndcg = 0;
    
    for i=1:querynum
        for j=1:rankernum-1
            for k=j+1:rankernum
                ndcg = 0;
                ranklist1 = ranklist(j,i,:);
                ranklist2 = ranklist(k,i,:);
                for m=1:item_num
                    if ranklist1(1,1,m) == ranklist2(1,1,m)
                        ndcg = ndcg + log(2)/log(m+1);
                    end
                end
                ndcglist(j,k) = ndcg;
                ndcglist(k,j) = ndcg;
            end
        end
        ndcgrank = sum(ndcglist,2);
        [~,ndcgrank] = sort(-ndcgrank);
        result(i,:) = rank(ndcgrank(1),i,:);
    end

        % save the .mat file which consist of the rank result
    save('D:\Code of RA\Preflib\results\ice-cream\rank-based\rank-result-ice-cream-PostNDCG.mat', 'result');
    
    end_time = toc(start_time);

    calculagraph(times) = end_time;
end

% print each run time
for times = 1:num_runs
    fprintf('run %d time：%.8f seconds\n', times, calculagraph(times));
end

% average_time
average_time = mean(calculagraph);
fprintf('average run time：%.8f seconds\n', average_time);
        
        