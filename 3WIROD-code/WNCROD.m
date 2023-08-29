%%Three-way neighborhood characteristic region-based outlier detection (3WIROD) algorithm
%%Please refer to the following papers:
%%Zhang xianyong,Yuan Zhong, and Miao Duoqian.Outlier Detection Using Three-Way
%%Neighborhood Characteristic Regions and Corresponding Fusion Measurement[J].TKDE,2023.
%%Uploaded by Yuan Zhong on August 29, 2023. E-mail:yuanzhong2799@foxmail.com.
function MNOF=WNCROD(data,X_tem,lammda)
%%%input:
% data is data matrix without decisions, where rows for samples and columns for attributes.
% Numerical attributes should be normalized into [0,1].
% Nominal attributes be replaced by different integer values.
% X_tem denotes the selected condition subdata.
% lammda is a given parameter for the radius adjustment.
%%%output
% Multiple neighborhood outlier factor (MNOF)

[n,m]=size(data);
X=zeros(1,n);
X(X_tem)=1;

D1=m/3;
D2=m/2;
D3=0.9*m;

delta=zeros(1,m);
ID=all(data<=1);
delta(ID)=std(data(:,ID),1)./lammda;

Lower=zeros(m,n);
for col=1:m
    RM_tem=pdist2(data(:,col),data(:,col),'cityblock')<=delta(col);
    Lower_temp=min(max(1- RM_tem,repmat(X,n,1)),[],2);
    Lower(col,1:length(Lower_temp))=Lower_temp;
end

IB=repmat(X,m,1)-Lower;
NEB=min(IB,[],1);
NPB=IB-repmat(NEB,m,1);

n_X=sum(X);
weight=zeros(n_X,m);
for col=1:m
    RM_tem=pdist2(data(:,col),data(:,col),'cityblock')<=delta(col);
    weight_x=[];
    for i=1:n_X
        temp1=RM_tem(X_tem(i),:);
        weight_temp=1-(sqrt((sum(min(temp1,X)))/n_X));
        weight_x=[weight_x,weight_temp];
    end
    weight((1:length(weight_x)),col)=weight_x';
end

D_tem=zeros(n);
for col=1:m
    RM_tem=pdist2(data(:,col),data(:,col),'cityblock')<=delta(col);
    D_tem=D_tem+RM_tem;
end
NOM=m-D_tem;
X_OM=NOM(X_tem,X_tem);

NEB_num=zeros(n_X,m);
Lower_num=zeros(n_X,m);
NPB_num=zeros(n_X,m);
for col=1:m
    temp2=Lower(col,:);
    temp3=NPB(col,:);
    for i=1:n_X
        temp1=X_OM(i,:);
        NEB_num(i,col)=sum(min(NEB(X_tem),temp1<=D1));
        NPB_num(i,col)=sum(min(temp3(X_tem),temp1>=D2));
        Lower_num(i,col)=sum(min(temp2(X_tem),temp1>=D3));
    end
end
NDF=(NEB_num+Lower_num+NPB_num)./n_X;
MNOF=mean(NDF.*weight,2);
end
