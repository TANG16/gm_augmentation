% STRATIFIEDLC generate stratified learning curve indexes
% INPUT:
%        labels - binary labels used for classification size N
% OUTPUT:
%        lc     - matrix containing indexes for trainingset
%                 size [N x number of training set sizes]
%        test   - (OPTIONAL) Logical containing indexes for 20% test set
%                 size N
%                 If test set is requested, lc(test,:) will be zeros
%
% lc is constructed such that if an observation is included, it will remain
% included for all training set sizes.
% Try imagesc(lc) for clarification
%
% Martin C. Axelsen 150318
% maxe@dtu.dk

function [lc, test] = stratifiedLC(labels)
% if used with a fixed test set size
if nargout>1
    labelsorg = labels;
    split = 5;
    cv = cvpartition(labelsorg,'kfold',split);
    rix = randperm(5,1);
    trix = cv.training(rix);
    labels = labelsorg(trix);
    test = cv.test(rix);
else
    trix = ones(size(labels))==1;
end
sml = min(mean(labels==1),mean(labels==0));
cv = cvpartition(labels,'kfold',floor(sml*numel(labels)));
lc = zeros(numel(trix),cv.NumTestSets);
for n=1:cv.NumTestSets
    lc(trix,n) = cv.test(n);
end
lc = cumsum(lc,2) == 1;
end