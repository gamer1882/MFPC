function  tY= MSSVC_iY_cwj(X,iY,c1,c2,sigma,mt)
% X is data;
% Y is getted by Initialization.m
% Y must be 1,2,...k.
% Cluster till convergence.
%num=max(Y);% the number of clustes
%totalw=zeros(n,num);
%mt the iter number of recursive 
m=size(X,1);% the number of samples
n=size(X,2);% the dimension of samples
flag=0;
alliter=0;
py=iY;%the lable of samples

while flag==0 && alliter<30
    alliter=alliter+1;
    tY=py;
    L=unique(tY);
    num=length(L);
    wti=zeros(n,mt,num);
    CenterX=zeros(num,n);
    dis=zeros(m,num);
    for i=1:num
        Ai=X(tY==L(i),:);
        Bi=X(tY~=L(i),:);
        CenterX(i,:)=mean(Ai);    
        for t=1:mt
            if t==1
                wt=zeros(n,1);
            elseif norm(wt)~=0
                wt=wt/norm(wt);
            end
            Ai=Ai-Ai*(wt*wt');
            Bi=Bi-Bi*(wt*wt');
            wt=lagrange_dual(Ai,Bi,c1,c2,sigma);
            wti(:,t,i)=wt;
        end   
    end
    for i=1:num
        M=X*wti(:,:,i)-repmat(CenterX(i,:)*wti(:,:,i),m,1);
        for j=1:m
            dis(j,i)=norm(M(j,:));
        end
    end 
    [~,py]=min(dis,[],2);
    if getAC(py,tY)>0.9999
        flag=1;
    end
end
end

function wi = lagrange_dual(Ai,Bi,c1,c2,sigma)
% A is the sample data
% c1,c2 and sigma are paramater
% w0 is the initial value
X=[Ai;Bi];
[mBi,n]=size(Bi);
eBi = ones(mBi,1);
LB = zeros(mBi,1);
UB = c2*ones(mBi,1);
I = eye(n);

XX = X'*X;
meam_A = mean(Ai);
BarAi = Ai-meam_A;
BarBi = Bi-meam_A;
Si=BarAi'*BarAi;

% computer: min||(A-Am)w||,s.t.||w||=1.
% initial w0
H = BarAi'*BarAi;
[V,D]=eig(H);
[~,n]=min(diag(D));
w0=V(:,n);

som=1;
iter=0;
tol=0.001;
while som>tol && iter<20
    iter=iter+1;
    Fi=sign(w0'*XX*w0-1);    
    Di = diag(sign(BarBi*w0));    
    Gi=(I+c1*Si+sigma*Fi*XX)\BarBi'*Di;
    H=Di*BarBi*Gi;
    H=(H+H')/2;
    alpha=qpSOR(H ,0.7,c2,0.05);
    wi=Gi*alpha;
    som=norm(wi-w0);
    w0=wi;
end
end

