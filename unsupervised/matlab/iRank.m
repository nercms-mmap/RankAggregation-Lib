% for calculagraph
num_runs = 10;
% calculagraph
calculagraph = zeros(1, num_runs);

for times = 1:num_runs

    % full ranking
    start_time = tic;
    dataset_name = 'ice cream';

    sim = importdata('D:\Code of RA\Preflib\results\ice-cream\ice-cream.mat');
    rankernum = size(sim,1);
    querynum = size(sim,2);
    item_num = size(sim,3);

    [~,ranklist] = sort(-sim,3);
    [~,rank] = sort(ranklist,3); 
    % 
    topK = item_num;
    
    result = zeros(querynum,item_num);
    superviserank = zeros(rankernum-1,querynum,item_num);
    newsim = zeros(rankernum,querynum,item_num);
    
    for iteration=1:3
        newsim = sim * 0.9;
        for i=1:rankernum
            if i==1
                superviserank = rank(2:rankernum,:,:);
            elseif i==rankernum
                superviserank = rank(1:rankernum-1,:,:);
            else
                superviserank(1:i-1,:,:) = rank(1:i-1,:,:);
                superviserank(i:rankernum-1,:,:) = rank(i+1:rankernum,:,:);
            end
            Dscore = 1./superviserank;
            DscoreTotal = sum(Dscore,1);
            [~,Sresultlist] = sort(-DscoreTotal,3);
            [~,Sresult] = sort(Sresultlist,3);
            Sresult = reshape(Sresult,querynum,item_num);
            Sresultlist = reshape(Sresultlist,querynum,item_num);
            for k=1:querynum
                for l=1:topK
                    newsim(i,k,Sresultlist(k,l)) = newsim(i,k,Sresultlist(k,l)) + 0.1;
                end
            end
        end
        sim = newsim;
        [~,ranklist] = sort(-sim,3);
        [~,rank] = sort(ranklist,3);
    end
    
    finalsim = sum(sim,1);
    finalsim = reshape(finalsim,querynum,item_num);
    [~,result] = sort(-finalsim,2);
    [~,result] = sort(result,2);
    % save the .mat file which consist of the rank result
    save('D:\Code of RA\Preflib\results\ice-cream\rank-based\rank-result-ice-cream-iRank.mat', 'result');

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
        
        