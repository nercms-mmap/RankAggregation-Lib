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

dirlist = dir('../testres/test_block1_prediciton0*.mat');
fileNames=string(cellstr(char({dirlist.name})));
metr=[];
for i=1:25
    file = fileNames(i);
    for iteration = 0
        iter = num2str(iteration);
        filepath=strcat('../testres/',file);
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

    %     dist = 1-reshape(mean(scores),num_probe,num_gallery);
    %     fprintf('mean\n');
    %     dist = 1-reshape(median(scores),num_probe,num_gallery);
    %     fprintf('median\n');

        %% evaluation
        [CMC, map, r1_pairwise, ap_pairwise] = evaluation(dist', label_gallery, label_query, cam_gallery, cam_query);

        CMC = 100.*CMC;
        cmc_result = CMC/100;
        auc_result = 0.5*(2*sum(cmc_result) - cmc_result(1) - cmc_result(end))/(length(cmc_result)-1);

    %     save([savepath 'dist_result/' dataset_name '_' feature_name '_' metric_name '_dist.mat'],'dist');
    %     save([savepath 'CMC_result/' dataset_name '_' feature_name '_' metric_name '_CMC.mat'],'CMC');
    %     save([savepath 'CMC_result/' dataset_name '_' feature_name '_' metric_name '_mAP.mat'],'map');

        matname=strsplit(file,'.');
        matname=matname(1);
        epoch=extractAfter(matname,'epoch');
    
        row=zeros(1,8);
        row(1)=epoch;
        row(2:6)=CMC([1,5,10,15,20]);
        row(7)=map*100;
        row(8)=auc_result*100;
        
        metr=[metr;row];

        fprintf(filepath);
        fprintf('\nRank1,  Rank5, Rank10, Rank15, Rank20\n');
        fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n', CMC([1,5,10,15,20]));
        fprintf(' auc=');
        fprintf('%5.2f%%', auc_result*100);
        fprintf(' map=');
        fprintf('%5.2f%%', map*100);
        fprintf('\n\n');
        CMC_results = CMC;
        results(iteration+1,:)= [CMC([1,5,10,20]),auc_result,map];
    end
end

csvwrite('evaluation_result.csv',metr);