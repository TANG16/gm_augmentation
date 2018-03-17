% MAIN function for calling and plotting results from UCIaugment
% INPUT:
%       ds - data set name i.e. 'ionosphere', 'adult', 'liver_disorders'
%       gN - number of synthetic samples to generate for augmentation
%
% OUTPUT:
%  displays learning curves of baseline models and augmented model.
%  saves plot to ../results

function main(ds, gN)
addpath('./utils');
download_dataset(ds)
if ~exist(sprintf('../data/result_data/%s_%d.mat',ds,gN), 'file')
    UCIaugment(ds, gN);
end
%%
files = dir(sprintf('../data/result_data/%s*.mat',ds));
Nfiles = numel(files);
for n = 1:Nfiles
    gN(n) = str2double(files(n).name(numel(ds)+2:end-4));
    fd(n) = load(sprintf('../data/result_data/%s',files(n).name));
end
% sort gN
[gN,tix] = sort(gN);
fd = fd(tix);
% plot baseline models NB and LR
l(1) = plot(fd(1).trsize,1-mean(fd(1).tenb,2),'-r');
hold on
l(2) = plot(fd(1).trsize,1-mean(fd(1).telr,2),'-b');
cix = linspace(0.1,1,numel(fd));
ltxt = {'NB','LR'};
% plot augmented models
for n=1:Nfiles
    l(n+2) = plot(fd(n).trsize,1-mean(fd(n).telrx,2),'Color',[0.2 cix(n) 0.2]);
    % plot arrows alternating from left and right
    if mod(n,2)
        text(fd(n).trsize(end),1-mean(fd(n).telrx(end,:)),...
            ['  \leftarrow' num2str(gN(n))],'HorizontalAlignment','left',...
            'FontSize',12);
    else
        text(fd(n).trsize(end),1-mean(fd(n).telrx(end,:)),...
            [num2str(gN(n)) '\rightarrow  ' ],'HorizontalAlignment','right',...
            'FontSize',12);
    end
    ltxt{n+2} = sprintf('LR a. w. %d samples',gN(n));
end
% simplify legend if several model performances are plotted
if Nfiles > 3
    l = l([1, 2, 3, Nfiles+2]);
    ltxt = ltxt([1, 2, 3, Nfiles+2]);
end
legend(l,ltxt)
xlabel('Training data size (m)')
ylabel('Error')
ds_disp = [upper(ds(1)) ...
    char(ds(2:end).*(ds(2:end)~='_') + ' '.*(ds(2:end)=='_'))];
title(ds_disp)
if ~exist('../results','dir')
    mkdir('../results/');
end
print(sprintf('../results/%s_plot.png',ds),'-dpng');