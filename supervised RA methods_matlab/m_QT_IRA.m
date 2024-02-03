clear

sim_path = 'C:\Users\2021\Desktop\MovieLens\mlm-top20-test.mat';
rel_path = 'C:\Users\2021\Desktop\MovieLens\mlm-top20-test-rel.mat';
datasetname = 'MovieLens-1m';

sim = importdata(sim_path);
fprintf('Running %s\n', datasetname);

rel_label = importdata(rel_path);



fprintf('Running %s\n', datasetname);

maxK = 5;
maxT = 5;

for it_k = 5:maxK
    for it_T = 1:maxT
    K = it_k;
    iteration = it_T;

    error_rate = 0.02; % Interaction error rate
    fprintf('K = %d\n',K);
    
    rankernum = size(sim,1);
    querynum = size(sim,2);
    gallerynum = size(sim,3);
    [~,ranklist] = sort(-sim,3);
    [~,rank] = sort(ranklist,3);
    
    averageRank = sum(sim, 1);
    averageRank = reshape(averageRank,querynum,gallerynum);
    [~,pseudoRanklist] = sort(-averageRank,2);
    [~,pseudoRank] = sort(pseudoRanklist,2);
    
    feedtrue = zeros(querynum,gallerynum);
    feeded = zeros(querynum,gallerynum);
    weight = ones(querynum,rankernum);
    
    %get origin rank
    origin_sim = zeros(querynum,gallerynum);
    for i=1:querynum
        for j = 1:rankernum
            origin_sim(i,:) = origin_sim(i,:) + reshape(double(sim(j,i,:)) * weight(i,j),1,gallerynum);
        end
    end
    [~,origin_ranklist] = sort(-origin_sim,2);
    [~,origin_rank] = sort(origin_ranklist,2);
    total_ranklist = origin_ranklist;
    result_1 = [];


    

    for i = 1:iteration
        new_weight = zeros(querynum,rankernum);
        tic
        for q = 1:querynum
            % Qlabel = query_label(q);
            sed = 0;
            now_num = 1;
            while sed<K
                if feeded(q,total_ranklist(q,now_num)) == 0
                    sed = sed +1;
                    RT(sed) = total_ranklist(q,now_num);
                    feeded(q,total_ranklist(q,now_num)) = 1;
                end
                now_num = now_num + 1;
            end
            % RT_label = gallery_label(RT);
            % feedback_P = find(RT_label == Qlabel);
            RT_label = rel_label(q,RT);
            feedback_P = find(RT_label == 1);
            for j = 1:K
                if ismember(j,feedback_P)
                    if rand(1) > error_rate
                        feedtrue(q,RT(j)) = 200;
                    end
                else
                    if rand(1) > error_rate
                        feedtrue(q,RT(j)) = -200;
                    end
                end
            end
            feedback_P = find(feedtrue(q,:)==200);
            feedback_N = find(feedtrue(q,:)==-200);
            
            if size(feedback_P,2) > 0
                score_P = sim(:,q,feedback_P);
                score_N = sim(:,q,feedback_N);
                score_P = reshape(score_P,rankernum,size(feedback_P,2));
                score_N = reshape(score_N,rankernum,size(feedback_N,2));
                S_P = sum(score_P,2);
                S_P = S_P ./ size(feedback_P,2); 
                S_N = sum(score_N,2);
                S_N = S_N ./ size(feedback_N,2);
                if size(feedback_N,2) == 0
                   S = S_P;
                else
                   S = S_P - S_N;
                end
               
                new_weight(q,:) = new_weight(q,:) + reshape(S,1,rankernum);
            end
        end
        
        weight = weight .* 0.1 + new_weight .* 0.9;
        for j = 1:querynum
            weight(j,:) = weight(j,:) ./ max(weight(j,:));
        end
        new_sim = zeros(querynum,gallerynum);
        for j=1:querynum
            for k = 1:rankernum
                new_sim(j,:) = new_sim(j,:) + reshape(double(sim(k,j,:)) * weight(j,k),1,gallerynum);
            end
        end
        new_sim = new_sim + feedtrue;
        toc
        [~,total_ranklist] = sort(-new_sim,2);
        [~,total_rank] = sort(total_ranklist,2);
    
    end
    
    filename = sprintf('result-%s-QT-IRA-k%d-T%d.mat', datasetname,K,iteration);
    save(filename,'new_sim');
    end
end