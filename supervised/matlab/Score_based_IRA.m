clear


%market
% simpath = 'D:\RA_ReID\Topk_ReID\Market1501\market1501_6workers.mat';
% query_id_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-market1501-query_id-.mat';
% gallery_idtest_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-market1501-gallery_idtest-.mat';
% cam_gallery_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-market1501-gallery_camidstest-.mat';
% cam_query_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-market1501-query_camids-.mat';

%duke
% simpath = 'D:\RA_ReID\Topk_ReID\DukeMTMC-ReID\dukemtmcreid_6workers.mat';
% query_id_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-dukemtmcreid-query_id-.mat';
% gallery_idtest_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-dukemtmcreid-gallery_idtest-.mat';
% cam_gallery_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-dukemtmcreid-gallery_camidstest-.mat';
% cam_query_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-dukemtmcreid-query_camids-.mat';

%detected
% simpath = 'D:\RA_ReID\Topk_ReID\CHUK03_detected\cuhk03detected_6workers.mat';
% query_id_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03detected-query_id-.mat';
% gallery_idtest_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03detected-gallery_idtest-.mat';
% cam_gallery_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03detected-gallery_camidstest-.mat';
% cam_query_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03detected-query_camids-.mat';

%label
simpath = 'D:\RA_ReID\Topk_ReID\CUHK03_labeled\cuhk03labeled_6workers.mat';
query_id_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03labeled-query_id-.mat';
gallery_idtest_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03labeled-gallery_idtest-.mat';
cam_gallery_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03labeled-gallery_camidstest-.mat';
cam_query_path = 'D:\RA_ReID\Topk_ReID\label&cam\bdb-cuhk03labeled-query_camids-.mat';

sim = importdata(simpath);
%sim = importdata('D:\文件\毕设\RA算法\new_mat\cuhk03labeled-test-111100-1000.mat');
% ranker * query * gallery
% fprintf('cuhk03labeled-eu-test:bdb\n');
% config; 

query_label0 = importdata(query_id_path);
% query_label = query_label0(701:1400);
query_label = query_label0;
query_label = query_label';
gallery_label = importdata(gallery_idtest_path);
cam_gallery = importdata(cam_gallery_path);
cam_gallery = cam_gallery';
cam_query0 = importdata(cam_query_path);
% cam_query = cam_query0(701:1400);
cam_query = cam_query0;
cam_query = cam_query';
%  701:1400  1115:2228   1685:3368
%query_label = importdata('D:\文件\毕设\RA算法\RA\dataset\cuhk03labeled\groundtruth\queryIDtest.mat');
%gallery_label = importdata('D:\文件\毕设\RA算法\RA\dataset\cuhk03labeled\groundtruth\galleryID.mat');
%cam_gallery = importdata('D:\文件\毕设\RA算法\RA\dataset\cuhk03labeled\groundtruth\galleryCAM.mat');
%cam_query1 = importdata('D:\文件\毕设\RA算法\RA\dataset\cuhk03labeled\groundtruth\queryCAMint.mat');
%cam_query = cam_query1(701:1400);
%cuhk03-1400;duke-2228;market-3368


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

feedtrue_G = zeros(querynum,gallerynum);
feeded_G = zeros(querynum,gallerynum);

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
            if feeded_G(q,total_ranklist(q,now_num)) == 0
                sed = sed +1;
                RT(sed) = total_ranklist(q,now_num);
                feeded_G(q,total_ranklist(q,now_num)) = 1;
            end
            now_num = now_num + 1;
        end
        RT_label = gallery_label(RT);
        scored_G = find(RT_label == Qlabel);
        for j = 1:K
            if ismember(j,scored_G)
                if rand(1) > error_rate
                    feedtrue_G(q,RT(j)) = 10;
                else
                    feedtrue_G(q,RT(j)) = -10;
                end
            else
                if rand(1) > error_rate
                    feedtrue_G(q,RT(j)) = -10;
                else
                    feedtrue_G(q,RT(j)) = 10;
                end
            end
        end
        scored_G = find(feedtrue_G(q,:)==10);
        if size(scored_G,2) > 1
            anno_G = sim(:,q,scored_G);
            anno_G = reshape(anno_G,rankernum,size(scored_G,2));
            std_w = std(anno_G,0,2);
            max_std = max(std_w);
            std_w = std_w ./ max_std;
            new_weight(q,:) = new_weight(q,:) + reshape(1./std_w,1,rankernum);
            total_weight = max(new_weight(q,:));
            new_weight(q,:) = new_weight(q,:) ./ total_weight;
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
    new_sim = new_sim + feedtrue_G;
    toc
    [~,total_ranklist] = sort(-new_sim,2);
    [~,total_rank] = sort(total_ranklist,2);

    %%% evaluation
    [CMC_result, map_result, ~, ~] = evaluation(total_rank', gallery_label, query_label, cam_gallery, cam_query);
    auc_result = 0.5*(2*sum(CMC_result) - CMC_result(1) - CMC_result(end))/(length(CMC_result)-1);
    result_1 = [result_1 ;CMC_result([1,5,10,20]).*100,auc_result, map_result];
    fprintf('iteration:%d r1:%.2f%% mAP:%.2f%%\n',i,100*CMC_result(1),100*map_result);
end

save('Score_based_cuhk03labeled_result.mat','total_rank');