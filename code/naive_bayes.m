% NAIVE_BAYES class for fitting, evaluating and generating samples with a
% naive Bayes classifier with normal distributed likelihood for continuous
% variables.
%
% CALL     nb = naive_bayes(Xtr,ytr,shared) to fit model.
% INPUT
%  Xtr:    training data [n x d]
%  ytr:    training labels [n x 1]
%  shared: bool, shared std across classes. Default, shared = 1.
% OUTPUT
%  nb:     object containing trained model
%
% CALL     [c,post] = nb.predict(Xte) to evaluate performance
% INPUT
%  Xte:    test data - if Xte == Xtr, output can be used for training error
% OUTPUT
%  c:      determined classes of each observation
% post:    posterior probabilities for each observation.
%
% CALL     [x,y] = nb.generate(N) to generate synthetic samples
% INPUT
%  N:      total number of samples to generate
% OUTPUT
%  x:      generated samples [N x d]
%  y:      generated labels [N x 1] - classes are stratified in accordance 
%          with the training data
%
% maxe 2017

classdef naive_bayes
    properties
        X   % Training data input
        y   % Training label input
        m   % mean training data [class 0;class 1]
        st  % std training data [class 0;class 1] same for both classes if shared==1
    end
    methods
        % FIT MODEL
        function obj=naive_bayes(X,y,shared)
            if exist('shared','var') && (shared == 0)
                % individual std per class
                obj.st(1,:) = std(X(y==0,:),[],1)+eps; % eps, small constant to avoid dividing by 0
                obj.st(2,:) = std(X(y==1,:),[],1)+eps;
            else
                % shared std across classes
                comstd = (sum(y==0)*std(X(y==0,:),[],1) + sum(y==1)*std(X(y==1,:),[],1))/numel(y);
                obj.st(1,:) = comstd+eps; % eps, small constant to avoid dividing by 0
                obj.st(2,:) = comstd+eps;
            end
            % mean for both classes
            obj.m(1,:) = mean(X(y==0,:),1);
            obj.m(2,:) = mean(X(y==1,:),1);
            obj.X = X;
            obj.y = y;
        end
        % PREDICT
        function [c,post] = predict(obj,Xte)
            % p(x|y)
            pxy0 = 1./(prod(obj.st(1,:))*sqrt(2*pi))*exp(sum((-(Xte-obj.m(1,:)).^2)./(2*obj.st(1,:).^2),2));
            pxy1 = 1./(prod(obj.st(2,:))*sqrt(2*pi))*exp(sum((-(Xte-obj.m(2,:)).^2)./(2*obj.st(2,:).^2),2));
            % p(x)
            px0 = sum(obj.y==0);
            px1 = sum(obj.y==1);
            % p(y|x)
            post = (prod(pxy1,2)*px1)./(prod(pxy1,2)*px1 + prod(pxy0,2)*px0);
%             c = post>0.5;
            c = log((prod(pxy1,2)*px1)./(prod(pxy0,2)*px0))>0;
        end
        % GENERATE
        function [x,y] = generate(obj,N)
            % preallocate
            D = size(obj.X, 2);
            x = zeros(N, D);
            % identify labels and permute order
            unq = unique(obj.y);
            rix = randperm(numel(unq));
            unq = unq(rix);
            % number observations in each class - respects existing class balance
            Nc = round(hist(obj.y,numel(unq))/numel(obj.y)*N);
            Nc = Nc(rix);
            Nc = [Nc(1:end-1), N-sum(Nc(1:end-1))];
            Ncs = cumsum([0, Nc]);           
            % loop over classes
            for ix = 1:numel(unq)
                % samples from normal distribution with params from naive bayes
                x(Ncs(ix)+1:Ncs(ix + 1), :) = randn(Nc(ix), D).*repmat(obj.st(rix(ix),:),Nc(ix),1) + repmat(obj.m(rix(ix),:), Nc(ix), 1);
                % labels
                y(Ncs(ix)+1:Ncs(ix + 1)) = unq(ix);
            end
            indx = randperm(N);
            x = x(indx,:);
            y = y(indx)';
        end
    end
end