% UCIAUGMENT Function for evaluating augmentation paradigme on UCI datasets
% INPUT:
%       ds    - name of dataset e.g. 'ionosphere'
%       gN    - number of generated samples e.g. 100
%       trint - training size interval for learning curve investigation
%               variable can be contained in data file
%
% OUTPUT
%  saved file with performance of models and meta information
%  saves to folder ../data/result_data/
%
% Note:
% Calls class naive_bayes, which must be in cwd.
%
% maxe 2018

function UCIaugment(ds, gN, trint)
if nargin<3
    load(sprintf('../data/%s.mat',ds),'dataorg','labelorg','trint');
else
    load(sprintf('../data/%s.mat',ds),'dataorg','labelorg');
end

%% Parameters which may be adjusted, but will increase computation time
% number of cross-validation repetitions
reps = 100;
% number of training set sizes to investigate (limited by stratification and
% available training data
lcinc = 10;

%%
fprintf('%s gN: %d rep: %d\n',ds,gN,reps);

% remove variables with zero variance
dataorg = dataorg(:,var(dataorg,[],1) ~= 0);
sml = min(mean(labelorg==1),mean(labelorg==0));
% check lcinc
lcincmax = (trint(2)-trint(1)+1)*sml;
if  lcinc > lcincmax
    lcinc = lower(lcincmax);
    warning(['Cannot create that amount of learning curve stratified '...
        'indeces from data.\n%9slcinc has been reset to: %d'],'',lcinc);
end
%% Preallocation of test error matrices
tenb = zeros(lcinc,reps);
telr = zeros(lcinc,reps);
telrx = zeros(lcinc,reps);

%%
start = tic;
% supress warnings from logistic regression fit
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:glmfit:PerfectSeparation');
warning('off','stats:glmfit:IllConditioned');
for rep = 1:reps
    fprintf('Rep no. %d / %d, time: %.3f\n',rep,reps,toc(start));
    rng(rep);
    % get indeces for stratified learningcurve and test set
    % [sensitive to rng seed]
    [lc, test_index] = stratifiedLC(labelorg);
    % enforce set training interval (from trint)
    lc = lc(:,logical((sum(lc,1)>=trint(1)) .* (sum(lc,1)<=trint(2))));
    % subsample learning curve to only include 10 increments
    lc = lc(:,round(linspace(1,size(lc,2),lcinc)));
    %
    X_test = dataorg(test_index, :);
    y_test = labelorg(test_index);
    %%
    trsize = sum(lc,1); % saved to output
    for ix = 1:size(lc,2)
        %% extract data for learning curve
        X_train = dataorg(lc(:,ix),:);
        y_train = labelorg(lc(:,ix));
        %% fit data
        nb = naive_bayes(X_train,y_train);
        lr = glmfit(X_train, y_train,'binomial');
        %% Generate data
        [genx, labx] = nb.generate(gN);
        indx = randperm(size(X_train,1)+gN);
        % augment training data
        Xx = [X_train; genx];
        Xx = Xx(indx,:);
        yx = [y_train; labx];
        yx = yx(indx);
        % fit augmented data
        lrx = glmfit(Xx, yx,'binomial');
        %% predict and evaluate
        tenb(ix, rep) = mean(nb.predict(X_test)==y_test);
        telr(ix, rep) = mean(glmval(lr, X_test, 'logit')>0.5==y_test);
        telrx(ix, rep) = mean(glmval(lrx, X_test, 'logit')>0.5==y_test);
    end
end
fprintf('Saving... time elapsed %.3f\n',toc(start));

%%
if ~exist('../data/result_data/','dir')
    disp('Creating directory ../data/result_data/')
    mkdir('../data/result_data');
end
svstr = sprintf('../data/result_data/%s_%d.mat',ds,gN);
save(svstr,'tenb','telr','telrx','trsize');
fprintf('Analysis results saved as %s\n',svstr)
