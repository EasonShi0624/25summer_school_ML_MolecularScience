%% chenjun, 2016/3/3
%% njuchenjun@gmail.com

clear
%% load ab initio data
xy=load('oh3-abinitio.txt');
x0=xy(:,1:6);
y0=xy(:,7);
%
xy=load('z1');
x0=[x0;xy(:,1:6)];
y0=[y0;xy(:,7)];
xy=load('z2');
x0=[x0;xy(:,1:6)];
y0=[y0;xy(:,7)];
% select some points for gaussian process
x=x0;y=y0;
nuse=800;
ntot=size(x,1);
ix=zeros(ntot,1);
while sum(ix)<nuse
    j=rand()*ntot;
    j=floor(j);
    j=min(j,ntot); j=max(j,1);
    ix(j)=1;
end

ix=zeros(ntot,1);
ix(1:2:16814)=1;
ix(16815:2:end)=1;

x=x(ix==1,:);
y=y(ix==1,1);

%xtmp=load('output2');
%x=xtmp(:,1:6);
%y=xtmp(:,7);
%clear xtmp

%%
%     nadd=0;
%     for i=1:size(yt,1)
%         if yt(i)<0 && abs(ys(i)-yt(i))>0.05
%             nadd=nadd+1;
%             ix(i)=1;
%         elseif yt(i)<0.5 && abs(ys(i)-yt(i))>0.1
%             nadd=nadd+1;
%             ix(i)=1;
%         elseif yt(i)<1 && abs(ys(i)-yt(i))>0.15
%             nadd=nadd+1;
%             ix(i)=1;
%         elseif yt(i)<1.5 && abs(ys(i)-yt(i))>0.2
%             nadd=nadd+1;
%             ix(i)=1;
%         end
%     end
%     nadd

%% training with gaussian process
ndim=size(x,2); ntot=size(x,1);

%xhandle='normal';
xhandle='reverse';
%xhandle='log';
if strcmp(xhandle,'reverse')==1
    x=x.\1;
elseif strcmp(xhandle,'log')==1
    x=log(x);
end

meanfunc = @meanZero;
covfunc = {@covMaternard, 5};
likfunc = @likGauss;
inffunc = @infExact;

hyp.cov=zeros(7,1);

hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -500, inffunc, meanfunc, covfunc, likfunc, x, y);
exp(hyp.lik)
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);

hyp.cov

%% calculate potential energy
tic
xt=x0; yt=y0;
if strcmp(xhandle,'reverse')==1
    xt=xt.\1;
elseif strcmp(xhandle,'log')==1
    xt=log(xt);
end

ys=gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xt);

rmse=sqrt(mse(yt-ys))*1000;
fprintf('RMSE = %16.8f meV\n',rmse);
toc
% %% plot errors
figure1 = figure(1);
clf(figure1);
axes1 = axes('Parent',figure1,'FontSize',16);
xlim(axes1,[-1 5]);
ylim(axes1,[-1 1]);
box(axes1,'on');
hold(axes1,'all');
plot(yt,ys-yt,'Marker','x','LineStyle','none');
xlabel('Potential Energy / eV','FontSize',16);
ylabel('V(Gaussian-Process) - V({\itab initio}) / eV','FontSize',16);


%% save GPinput.txt for Fortran
fid=fopen('GPinput.txt','w');

fprintf(fid,'%s\n','# fuctions');
fprintf(fid,'%s\n',char(inffunc));
fprintf(fid,'%s\n',char(meanfunc));
fprintf(fid,'%s\n',char(covfunc{1}));
if strcmp(char(covfunc{1}),'covMaternard') || strcmp(char(covfunc{1}),'covMaterniso')
    fprintf(fid,'%d\n',covfunc{2});
end
fprintf(fid,'%s\n',char(likfunc));

fprintf(fid,'%s\n','# ndim and ndata');
fprintf(fid,'%d %d\n',size(x,2),size(x,1));
fprintf(fid,'%s\n','# hyperparameter mean');
if ~strcmp(char(meanfunc),'meanZero')
    for i=1:size(hyp.mean,1)
        fprintf(fid,'%28.20e\n',hyp.mean(i));
    end
end
fprintf(fid,'%s\n','# hyperparameter cov');
for i=1:size(hyp.cov,1)
    fprintf(fid,'%28.20e\n',hyp.cov(i));
end
fprintf(fid,'%s\n','# hyperparameter lik');
for i=1:size(hyp.lik,1)
    fprintf(fid,'%28.20e\n',hyp.lik(i));
end
fprintf(fid,'%s\n','# training data');
fprintf(fid,'%s\n',xhandle);
for i=1:size(x,1)
    for j=1:size(x,2)
        if strcmp(xhandle,'reverse')
            fprintf(fid,'%13.7f',1/x(i,j));
        elseif strcmp(xhandle,'normal')
            fprintf(fid,'%13.7f',x(i,j));
        elseif strcmp(xhandle,'log')
            fprintf(fid,'%13.7f',exp(x(i,j)));
        end
    end
    fprintf(fid,'%14.8f\n',y(i));
end

fclose(fid);

%%
fid=fopen('GPtest.txt','w');
for i=1:size(xt,1)
    for j=1:size(xt,2)
        if strcmp(xhandle,'reverse')
            fprintf(fid,'%13.7f',1/xt(i,j));
        elseif strcmp(xhandle,'normal')
            fprintf(fid,'%13.7f',xt(i,j));
        elseif strcmp(xhandle,'log')
            fprintf(fid,'%13.7f',exp(xt(i,j)));
        end
    end
    fprintf(fid,'%14.8f\n',yt(i));
end

fclose(fid);
