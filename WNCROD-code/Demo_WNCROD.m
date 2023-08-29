clc;
clear
format shortG;

load Example.mat

Dataori=Example;

trandata=Dataori;
trandata(:,2:3)=normalize(trandata(:,2:3),'range');

X_tem=[1,2,5,6];
lammda=1;

out_scores=WNCROD(trandata,X_tem,lammda)

