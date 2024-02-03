clear

% addpath(genpath('D:\文件\毕设\RA算法\order_worker\Sim\'));

%duke
% sim_path = 'D:\RA_ReID\ReID_Dataset\DukeMTMC-ReID\test\dukemtmcreid_6workers.mat';
% query_label_path = 'D:\RA_ReID\ReID_Dataset\DukeMTMC-ReID\label&cam\bdb-dukemtmcreid-query_id-.mat';
% gallery_label_path = 'D:\RA_ReID\ReID_Dataset\DukeMTMC-ReID\label&cam\bdb-dukemtmcreid-gallery_idtest-.mat';
% cam_gallery_path = 'D:\RA_ReID\ReID_Dataset\DukeMTMC-ReID\label&cam\bdb-dukemtmcreid-gallery_camidstest-.mat';
% cam_query_path = 'D:\RA_ReID\ReID_Dataset\DukeMTMC-ReID\label&cam\bdb-dukemtmcreid-query_camids-.mat';
% datasetname = 'duke';

%detected
% sim_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_detected\test\cuhk03detected_6workers.mat';
% query_label_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_detected\label&cam\bdb-cuhk03detected-query_id-.mat';
% gallery_label_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_detected\label&cam\bdb-cuhk03detected-gallery_idtest-.mat';
% cam_gallery_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_detected\label&cam\bdb-cuhk03detected-gallery_camidstest-.mat';
% cam_query_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_detected\label&cam\bdb-cuhk03detected-query_camids-.mat';
% datasetname = 'detected';

%label
sim_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\test\cuhk03labeled_6workers.mat';
query_label_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-query_id-.mat';
gallery_label_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-gallery_idtest-.mat';
cam_gallery_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-gallery_camidstest-.mat';
cam_query_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-query_camids-.mat';
datasetname = 'label';

%market
% sim_path = 'D:\RA_ReID\ReID_Dataset\Market1501\test\market1501_6workers.mat';
% query_label_path = 'D:\RA_ReID\ReID_Dataset\Market1501\label&cam\bdb-market1501-query_id-.mat';
% gallery_label_path = 'D:\RA_ReID\ReID_Dataset\Market1501\label&cam\bdb-market1501-gallery_idtest-.mat';
% cam_gallery_path = 'D:\RA_ReID\ReID_Dataset\Market1501\label&cam\bdb-market1501-gallery_camidstest-.mat';
% cam_query_path = 'D:\RA_ReID\ReID_Dataset\Market1501\label&cam\bdb-market1501-query_camids-.mat';
% datasetname = 'market';




sim = importdata(sim_path);
%sim = importdata('D:\文件\毕设\RA算法\order_worker\Sim7\cuhk03labeled-test-1111111-1.mat');
% ranker * query * gallery
% fprintf('cuhk03labeled-eu-test:bdb\n');
fprintf('Running %s\n', datasetname);
% config; 

query_label0 = importdata(query_label_path);
% query_label = query_label0(701:1400);
query_label = query_label0;
query_label = query_label';
gallery_label = importdata(gallery_label_path);
cam_gallery = importdata(cam_gallery_path);
cam_gallery = cam_gallery';
cam_query0 = importdata(cam_query_path);
% cam_query = cam_query0(701:1400);
cam_query = cam_query0;
cam_query = cam_query';
%  701:1400  1115:2228   1685:3368

iteration = 1; % Number of interaction rounds
K = 5; % Interaction per round
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
        origin_sim(i,:) = origin_sim(i,:) + reshape(sim(j,i,:) * weight(i,j),1,gallerynum);
    end
end
[~,origin_ranklist] = sort(-origin_sim,2);
[~,origin_rank] = sort(origin_ranklist,2);
total_ranklist = origin_ranklist;
result_1 = [];

%%% evaluation
[CMC_result, map_result, ~, ~] = evaluation(origin_rank', gallery_label, query_label, cam_gallery, cam_query);
auc_result = 0.5*(2*sum(CMC_result) - CMC_result(1) - CMC_result(end))/(length(CMC_result)-1);
result_1 = [CMC_result([1,5,10,20]).*100,auc_result, map_result];
fprintf('original r1:%.2f%% mAP:%.2f%%\n',100*CMC_result(1),100*map_result);


for i = 1:iteration
    new_weight = zeros(querynum,rankernum);
    tic
    for q = 1:querynum
        Qlabel = query_label(q);
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
        RT_label = gallery_label(RT);
        feedback_P = find(RT_label == Qlabel);
        for j = 1:K
            if ismember(j,feedback_P)
                if rand(1) > error_rate
                    feedtrue(q,RT(j)) = 10;
                end
            else
                if rand(1) > error_rate
                    feedtrue(q,RT(j)) = -10;
                end
            end
        end
        feedback_P = find(feedtrue(q,:)==10);
        feedback_N = find(feedtrue(q,:)==-10);
        
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
            new_sim(j,:) = new_sim(j,:) + reshape(sim(k,j,:) * weight(j,k),1,gallerynum);
        end
    end
    new_sim = new_sim + feedtrue;
    toc
    [~,total_ranklist] = sort(-new_sim,2);
    [~,total_rank] = sort(total_ranklist,2);

    %%% evaluation
    [CMC_result, map_result, ~, ~] = evaluation(total_rank', gallery_label, query_label, cam_gallery, cam_query);
    auc_result = 0.5*(2*sum(CMC_result) - CMC_result(1) - CMC_result(end))/(length(CMC_result)-1);
    result_1 = [result_1 ;CMC_result([1,5,10,20]).*100,auc_result, map_result];
    fprintf('iteration:%d r1:%.2f%% mAP:%.2f%%\n',i,100*CMC_result(1),100*map_result);
end

