% for calculagraph
num_runs = 10;
% calculagraph
calculagraph = zeros(1, num_runs);

for times = 1: num_runs
    % Full ranking
    start_time = tic;
    
    dataset_name = 'ice cream';

    sim = importdata('D:\Code of RA\Preflib\results\ice-cream\ice-cream.mat');
    
    rankernum = size(sim,1);
    querynum = size(sim,2);
    item_num = size(sim,3);
    [~,rank] = sort(-sim,3);
    [~,rank] = sort(rank,3);
    
    res = zeros(querynum,item_num);
    condorcet_score = zeros(item_num,item_num);
    total_score = zeros(querynum,item_num);
    
    for i = 1:rankernum
        for n = 1:item_num
            for m = 1:item_num
                g = rank(i,1,n);
                if g < rank(i,1,m)
                    condorcet_score(n,m) = condorcet_score(n,m) + 1;
                else 
                    condorcet_score(n,m) = condorcet_score(n,m) + 0;
                end
            end
        end
    end

    for i =1:querynum
        for n = 1:item_num
            for m =1:item_num
                if condorcet_score(n,m) > 1
                total_score(i,n)= total_score(i,n) + 1;
                end
            end
        end
    end

    [~,res] = sort(-total_score,2);
    [~,res] = sort(res,2);
    
    res = reshape(res,querynum,item_num);
    
    % save the .mat file which consist of the rank result
    save('D:\Code of RA\Preflib\results\ice-cream\rank-based\rank-result-ice-cream-Condorcet.mat', 'res');
    
    end_time = toc(start_time);

    % calculagraph
    calculagraph(times) = end_time;
end

% print each run time
for times = 1:num_runs
    fprintf('run %d time：%.8f seconds\n', times, calculagraph(times));
end

% average_time
average_time = mean(calculagraph);
fprintf('average run time：%.8f seconds\n', average_time);   
        