function Y= Initialization(X,k,knn)
% NNG INITIALIZATION initialize the input X to get the cluster Y.
% Y is the pre-predicted cluster label.
% Idea: using knn to get clusters.
% k: the number of clusters you want.
% knn: the parameter in knn algorithm.
m=size(X,1);
Y=zeros(m,1);
book=sparse(m,m); % distance matrix
cnt=sparse(m,m); % connection matrix
% compute the distance matrix %
d=1;
for i=1:m-1
    d=d+1;
    j=d;
    while j~=m+1
        book(i,j)=norm(X(i,:)-X(j,:));
        j=j+1;
    end
end
% compute the connection matrix %
for i=1:m
    dis=book(i,2:m);
    for j=1:i-1
        dis(1,j)=book(j,i);
    end
    [a,b]=sort(dis);
    b=b(1:knn);
    b(b>=i)=b(b>=i)+1;
    cnt(i,b)=1;
end
% get clusters by the connection %
tmpY=GetCluster(cnt);
num=max(tmpY);
% combination if number is greater than expected %
if num>k
    disp('Combination');
    Y=Combination(book,k,tmpY);
elseif num<k
% decomposition if number if lesser than expected %
    disp('Decomposition');
    Y=Decomposition(book,cnt,k,tmpY);
else
    Y=tmpY;
end
end


function tY=Combination(book,k,tmpY)
m=max(tmpY); % m must be greater than k
clusterbook=sparse(m,m); %upper triangle matrix
for i=1:m-1
    for j=i+1:m
        cluster1=find(tmpY==i);
        cluster2=find(tmpY==j);        
        dis=GetDistance(book,cluster1,cluster2);
        if dis==0
            clusterbook(i,j)=1e-10;
        else
            clusterbook(i,j)=dis;
        end
    end
end
denum=m-k;
[row,col,val]=find(clusterbook);
[tmp,order]=sort(val);
Y=tmpY;
i=1;
while m~=k
    if Y(find(Y==row(order(i)),1))~=Y(find(Y==col(order(i)),1))
        m=m-1;
        Y(Y==col(order(i)))=Y(find(Y==row(order(i)),1));
       % num=CheckLabel(Y)
    end    
    i=i+1;
end
tY=CorrectLabel(Y);
end

function num=CheckLabel(Y)
% count the label number of Y
m=length(Y);
lab=[];
num=0;
for i=1:m
    if isempty(find(lab==Y(i), 1))
        lab=[lab;Y(i)];
        num=num+1;
    end
end
end

function tY=CorrectLabel(Y)
% Correct label from 1, 2, 3, ... , to max
n=length(Y);
m=max(Y);
tY=Y;
for i=1:m
    while isempty(find(tY==i, 1))
        tY(tY>i)=tY(tY>i)-1;
    end
    if i>=max(tY)
        break;
    end
end
end

function val=GetDistance(book,v1,v2)
% hausdorff distance
m1=length(v1);m2=length(v2);
val2=-inf;
for i=1:m1
    val1=inf;
    for j=1:m2
        if val1>book(v1(i),v2(j))
            val1=book(v1(i),v2(j));
        end
    end
    if val2<val1
        val2=val1;
    end
end
val3=-inf;
for i=1:m1
    val1=inf;
    for j=1:m2
        if val1>book(v1(i),v2(j))
            val1=book(v1(i),v2(j));
        end
    end
    if val3<val1
        val3=val1;
    end
end
val=max(val2,val3);
end


function Y=GetCluster(cnt)
m=size(cnt,1);
connection=cnt+cnt';
connection=triu(connection)+speye(m); % symmetric connection matrix
labelrecord=sparse(m,m);
Y=zeros(m,1);
label=1;
for i=1:m
    a=find(connection(i,1:m)~=0);
    val=find(Y(a,1)~=0); 
    if ~isempty(val)
        what=CountLabel(Y,a);
        if length(what)>1
            for j=2:length(what)
                Y(Y==what(j))=what(1);
            end
        end
        Y(a,1)=Y(a(val(1)),1);        
    else
        Y(a,1)=label;
        label=label+1;
    end
end
Y=CorrectLabel(Y);
end

function what=CountLabel(Y,order)
% count what labels in Y of the order
what=[];
n=length(order);
for i=1:n
    if isempty(find(what==Y(order(i)),1)) && Y(order(i))~=0
        what=[what;Y(order(i))];
    end
end
end

function Y=Decomposition(book,cnt,k,tmpY)
% book: symmetric matrix of distances between two samples
% cnt: matrix of connection between two samples by knn
% max(tmpY) must be less than k
m=length(tmpY);
cnt=cnt+cnt';
cnt=triu(cnt);
[row,col]=find(cnt);
n=length(row);
val=zeros(1,n);
for i=1:n
    val(i)=book(row(i),col(i));
end
[tmp,order]=sort(val,'descend');
for i=1:n
    cnt(row(order(i)),col(order(i)))=0;
    cnt(col(order(i)),row(order(i)))=0;
    Y=GetCluster(cnt);
    if max(Y)==k
        break;
    end
end
end


