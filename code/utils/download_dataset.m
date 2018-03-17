% DOWNLOAD_DATASET download and save data sets from UCI database
%   Downloads data sets from UCI repository, extracts continuous and binary
%   variables, as done in [Ng and Jordan 2002] and [Xue and Titterington
%   2008] (note, gender is not included for adult data set).
% INPUT: 
%       ds       - name of dataset ['ionosphere', 'adult', 
%                  'liver_disorders'].
%
% OUTPUT: [ds].mat file (e.g. adult.mat) containing:
%       dataorg  - matrix size [observations x variables].
%       labelorg - vector containing binary labels for classification task
%                  as suggested by UCI.
%       trint    - Interval of training set sizes to investigate (same
%                  interval as used in [Axelsen et al. 2018].
%
% Martin C. Axelsen 150318
% maxe@dtu.dk

function download_dataset(ds)
if exist(sprintf('../data/%s.mat',ds),'file')
    disp('File already exists in data folder')
else
    if ~exist('../data','dir')
        mkdir('../data');
    end
    switch ds
        case 'ionosphere'
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data';
            fname = '../data/ionosphere_data.txt';
            websave(fname,url);
            tbl = readtable(['../data/ionosphere_data.txt']);
            dataorg = table2array(tbl(:,1:end-1));
            labelorg = double(strcmp(table2cell(tbl(:,end)),'g'));
            trint = [35 281];
            save ../data/ionosphere.mat dataorg labelorg trint
        case 'adult'
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data';
            fname = '../data/adult_data.txt';
            websave(fname,url);
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test';
            fname = '../data/adult_test.txt';
            websave(fname,url);
            % keep only continous variables and concatenate data and test
            % age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
            cb_ix = [1 3 5 11 12 13];
            tbl_data = readtable('../data/adult_data.txt');
            tbl_test = readtable('../data/adult_test.txt');
%             tbl_data.Var10 = strcmp(tbl_data.Var10,'Male')*1;
%             tbl_test.Var10 = strcmp(tbl_test.Var10,'Male')*1;
            tbl = [tbl_data;tbl_test];
            dataorg = table2array(tbl(:,cb_ix));
            labelorg = double(strcmp(table2cell(tbl(:,end)),'>50K'));
            trint = [9 196];
            save ../data/adult.mat dataorg labelorg trint
        case 'liver_disorders'
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data';
            fname = '../data/liver_disorders_data.txt';
            websave(fname,url);
            tbl = readtable(['../data/liver_disorders_data.txt']);
            % remove dublicates as described in:
            % https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/noteDuplicates.txt
            dbls = ismember((1:size(tbl,1)),[86 318 150 176]);
            dataorg = table2array(tbl(~dbls,1:end-1));
            labelorg = table2array(tbl(~dbls,end))-1;
            trint = [8 273];
            save ../data/liver_disorders.mat dataorg labelorg trint
    end
end
