clear;
addpath(genpath('/dat01/fanxinyao/ext-CSRA/aggregate/cuhk03labeled/data'));
%% setting
train = 2; % train = 1, test = 2
results = [];
dataset_name = 'cuhk03labeled';
feature_name = '11111110101';
metric_name = '11111111';
savepath = '../evawres/';
datapath = '/dat01/fanxinyao/ext-CSRA/aggregate/cuhk03labeled/data/';


for iteration = 0
    iter = num2str(iteration);
    filepath=strcat('../testres/test_block1_prediciton0_epoch3000.mat');
    %% import data
    label_gallery = importdata([datapath,'labels_gallery.mat']);
    if train == 1
        label_query = importdata([datapath, 'labels_probe_Train.mat']);
%         abilities = importdata('C:\Users\yuyin\Desktop\train_block1_label.mat');
%         abilities = importdata('C:\Users\yuyin\Desktop\train_block1_prediciton.mat');
        abilities = importdata(['train_block1_prediciton' iter '.mat']);
%         abilities = importdata([module_path '\train_block1_label.mat']);
        scores = importdata([datapath,'Train.mat']);
    else
        label_query = importdata([datapath, 'labels_probe_Test.mat']);
%         abilities = importdata('C:\Users\yuyin\Desktop\test_block1_label.mat');
%         abilities = importdata('C:\Users\yuyin\Desktop\test_block1_prediciton.mat');
%         abilities = importdata(['test_block1_prediciton' iter '.mat']);
%         abilities = importdata('testres/test_block1_prediciton0_epoch1000.mat');
        abilities = importdata(filepath);
%         abilities = importdata([module_path '\test_block1_label.mat']);
        scores = importdata('../data/test.mat');
%         scores = importdata('C:\Users\yuyin\Desktop\Test.mat');
    end



    num_workers = size(scores,1);
    num_probe = size(scores,2);
    num_gallery  =size(scores,3);

    cam_gallery = importdata([datapath, 'testCAM.mat']);
    cam_query = importdata([datapath, 'queryCAM.mat']);
    cam_query = cam_query((num_probe+1):end,:);
    %% weighting
    weighted_scores = [];
    for j = 1:num_probe
        ability = [];
        for i = 1:num_workers
            ability(i,1) = abilities(j+(i-1)*num_probe,1);
        end
        ability_sum = sum(ability);
        ability = ability/ability_sum;
        score = scores(:,j,:);
        score = reshape(score,num_workers,num_gallery);
        [weight,ind] = sort(ability,'descend');
        new_scores = zeros(1,num_gallery);
        for i = 1:num_workers
            k = ind(i,1);
            weighting = weight(i,1)* score(k,:);
            new_scores = new_scores+weighting;
        end
        weighted_scores(j,:) = new_scores;
    end

%     weighted_scores1 = importdata('C:\Users\yuyin\Desktop\test_weighted_scores4.mat');
%     weighted_scores1 = reshape(weighted_scores1,158,316);
    dist = 1-weighted_scores;

    dist1 = 1-reshape(mean(scores),num_probe,num_gallery);
    dist2 = 1-reshape(median(scores),num_probe,num_gallery);

    %% evaluation
    [CMC, map, r1_pairwise, ap_pairwise] = evaluation(dist', label_gallery, label_query, cam_gallery, cam_query);

    CMC = 100.*CMC;
    cmc_result = CMC/100;
    auc_result = 0.5*(2*sum(cmc_result) - cmc_result(1) - cmc_result(end))/(length(cmc_result)-1);

    
    [CMC1, map1, r1_pairwise, ap_pairwise] = evaluation(dist1', label_gallery, label_query, cam_gallery, cam_query);

    CMC1 = 100.*CMC1;
    cmc_result1 = CMC1/100;
    auc_result1 = 0.5*(2*sum(cmc_result1) - cmc_result1(1) - cmc_result1(end))/(length(cmc_result1)-1);

    save([savepath 'dist_result/' dataset_name '_dist.mat'],'dist');
    save([savepath 'CMC_result/' dataset_name '_CMC.mat'],'CMC');
    save([savepath 'CMC_result/' dataset_name '_mAP.mat'],'map');


    fprintf('mean\n');
    fprintf('\nRank1, Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n', CMC1([1,5,10,15,20]));
    fprintf(' auc=');
    fprintf('%5.2f%%', auc_result1*100);
    fprintf(' map=');
    fprintf('%5.2f%%', map1*100);
    fprintf('\n\n');
    CMC_results1 = CMC1;
    results(iteration+1,:)= [CMC1([1,5,10,20]),auc_result1,map1];
    
    
    [CMC2, map2, r1_pairwise, ap_pairwise] = evaluation(dist2', label_gallery, label_query, cam_gallery, cam_query);

    CMC2 = 100.*CMC2;
    cmc_result2 = CMC2/100;
    auc_result2 = 0.5*(2*sum(cmc_result2) - cmc_result2(1) - cmc_result2(end))/(length(cmc_result2)-1);

%     save([savepath 'dist_result/' dataset_name '_dist.mat'],'dist');
%     save([savepath 'CMC_result/' dataset_name '_CMC.mat'],'CMC');
%     save([savepath 'CMC_result/' dataset_name '_mAP.mat'],'map');


    fprintf('median\n');
    fprintf('\nRank1, Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n', CMC2([1,5,10,15,20]));
    fprintf(' auc=');
    fprintf('%5.2f%%', auc_result2*100);
    fprintf(' map=');
    fprintf('%5.2f%%', map2*100);
    fprintf('\n\n');
    CMC_results2 = CMC2;
    results(iteration+1,:)= [CMC2([1,5,10,20]),auc_result2,map2];
end