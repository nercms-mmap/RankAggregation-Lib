% for calculagraph
num_runs = 10;
% calculagraph
calculagraph = zeros(1, num_runs);

dataset_name = 'ice cream';

for times = 1:num_runs
    % full ranking
    start_time = tic;

    sim = importdata('D:\Code of RA\Preflib\results\ice-cream\ice-cream.mat');
    
    ranker_num = size(sim,1);
    query_num = size(sim,2);
    item_num = size(sim,3);

    [~,rank] = sort(-sim,3);
    [~,rank] = sort(rank,3);

    res = zeros(query_num,item_num);

    borda_score = item_num - rank + 1;
    borda_score = sum(borda_score,1);
    
    [~,res] = sort(-borda_score,3);
    [~,res] = sort(res,3);

    res = reshape(res,query_num,item_num);
    
    % save the .mat file which consist of the rank result
    save('D:\Code of RA\Preflib\results\ice-cream\rank-based\rank-result-ice-cream-BordaCount.mat', 'res');
    
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