% for calculagraph
num_runs = 10;
% calculagraph
calculagraph = zeros(1, num_runs);

for times = 1:num_runs

    start_time = tic;
    dataset_name = 'ice cream';

    sim = importdata('D:\Code of RA\Preflib\results\ice-cream\ice-cream.mat');

    rankernum = size(sim,1);
    querynum = size(sim,2);
    item_num = size(sim,3);
    res = zeros(querynum,item_num);
    
    res = reshape(mean(sim),querynum,item_num);

    % save the .mat file which consist of the rank result
    % store score
    save('D:\Code of RA\Preflib\results\ice-cream\score-based\result-ice-cream-Mean.mat', 'res');

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