clear
% Load the data from the MAT file
data = importdata('C:\Users\2021\Desktop\SocialChoice\university-ranking-test.mat');
datasetname = 'university-ranking';
%参数
TopK = 10;

% Get the size of the data matrix
%[row, col, dim] = size(data);
rankernum = size(data,1);
querynum = size(data,2);
gallerynum = size(data,3);
[~,ranklists] = sort(data,3);

converged = zeros(querynum,rankernum);
new_w = zeros(querynum,rankernum);
w0 = ones(querynum,rankernum);
w0 = w0 ./ rankernum;

L = zeros(querynum,gallerynum);
for i=1:querynum
    for j = 1:rankernum
        L(i,:) = L(i,:) + reshape(double(data(j,i,:)) * w0(i,j),1,gallerynum);
    end
end
[~,origin_ranklist] = sort(-L,2);
[~,origin_rank] = sort(origin_ranklist,2);
[~,original_voters] = sort(-data,3);
% [~,original_voters] = sort(-data,3);

for q=1:querynum
    now_L =  origin_ranklist(q,:);
    now_L_rank = origin_rank(q,:);
    i = 0;
    allconverged = 0;
    pre_w = w0(q,:);
    now_w = zeros(1,rankernum);
    while allconverged == 0
        i = i + 1;
        allconverged = 1;
%         now_w = zeros(1,rankernum);
        for r=1:rankernum
           if  converged(q,r) == 0
               distance = 0;
               [~,V_ranklist] = sort(-data(r,q,:),3);
               V_ranklist = reshape(V_ranklist,1,gallerynum);
               [~,V_rank] = sort(V_ranklist,2);
%                V_ranklist = V_ranklist(1:30);
%                V_rank = V_rank(1:30);
               for j=1:TopK
%                    idx_V = find(V_rank == j);
                   idx_V = V_ranklist(j);
                   distance = distance + abs(j/TopK - (now_L_rank(idx_V)/gallerynum));
               end
               distance = distance / (TopK/2);
               now_w(1,r) = pre_w(1,r) + exp(-i * distance);
               if (now_w(1,r) - pre_w(1,r)) > 0.001
                   allconverged = 0;
               else
                   converged(q,r) = 1;
               end
           else
              now_w(1,r) = pre_w(1,r);
           end
        end
        new_L = zeros(1,gallerynum);
        for r=1:rankernum
%             L(q,:) = reshape(L(q,:) * now_w(1,r),1,gallerynum);
            new_L(1,:) = new_L(1,:) + reshape(double(data(r,q,:)) * now_w(1,r),1,gallerynum);
            pre_w(1,r) = now_w(1,r);
        end
%         [~,now_L] =  sort(-L(q,:),2);
        [~,now_L] =  sort(-new_L,2);
        [~,now_L_rank] = sort(now_L,2);
    end
    new_w(q,:) = now_w;
end

new_L = zeros(querynum,gallerynum);
for j=1:querynum
    for r=1:rankernum
        new_L(j,:) = new_L(j,:) + reshape(double(data(r,j,:)) * new_w(j,r),1,gallerynum);
    end
end

filename = sprintf('result-%s-DIBRA.mat', datasetname);
save(filename,'new_L');



% [~,total_ranklist] = sort(-new_L,2);
% [~,total_rank] = sort(total_ranklist,2);
% result_1 = [];
% query_label0 = importdata('C:\Users\29984\Desktop\new_worker\order_fea\bdb-cuhk03detected-query_id-.mat');
% query_label = query_label0';
% gallery_label = importdata('C:\Users\29984\Desktop\new_worker\order_fea\bdb-cuhk03detected-gallery_idtest-.mat');
% cam_gallery = importdata('C:\Users\29984\Desktop\new_worker\order_fea\bdb-cuhk03detected-gallery_camidstest-.mat');
% cam_gallery = cam_gallery';
% cam_query0 = importdata('C:\Users\29984\Desktop\new_worker\order_fea\bdb-cuhk03detected-query_camids-.mat');
% cam_query = cam_query0';
% [CMC_result, map_result, ~, ~] = evaluation(total_rank', gallery_label, query_label, cam_gallery, cam_query);
% auc_result = 0.5*(2*sum(CMC_result) - CMC_result(1) - CMC_result(end))/(length(CMC_result)-1);
% result_1 = [result_1 ;CMC_result([1,5,10,20]).*100,auc_result, map_result];
% fprintf('r1:%.2f%% mAP:%.2f%%\n',100*CMC_result(1),100*map_result);


