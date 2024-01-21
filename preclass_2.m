function [xx1,xx2]=preclass_2(X,Y,Z)
% [X]=preclass_1(X);

% Z(t,:)=Z;
train_x1=(reshape(X',1024,50*size(X,1)))';
train_x1=mapminmax(train_x1,0,1);
train_x2=(reshape(Y',1024,50*size(Y,1)))';
train_x2=mapminmax(train_x2,0,1);
train_x3=(reshape(Z',1024,50*size(Z,1)))';
train_x3=mapminmax(train_x3,0,1);
% s=train_x3;

for i=1:5
    p1=X(i,:); %第i的所有元素
    p2=Y(i,:);
    tr1=(reshape(p1',10,5*1024))';%共轭转置，
    tr2=(reshape(p2',10,5*1024))';
    tr1=(reshape(tr1,50,1024));
    tr2=(reshape(tr2,50,1024));
    tr1=mapminmax(tr1,0,1);
    tr2=mapminmax(tr2,0,1);
%     for j=1:5
        p3=Z(i,:);
        tr3=(reshape(p3',10,5*1024))';
        tr3=(reshape(tr3,50,1024));
        tr3=mapminmax(tr3,0,1);
        tr11(1+(i-1)*50:i*50,:)=tr1;
        tr22(1+(i-1)*50:i*50,:)=tr2;
        tr33(1+(i-1)*50:i*50,:)=tr3;
end
for k=1:30
tt=randperm(5);
tp{k}=tt;
for p=1:5
tu(1+(p-1)*50:p*50,:)=tr22(1+(tt(p)-1)*50:tt(p)*50,:);
end
Hx =1 ./ (1 + exp(- tu));
Hz =1 ./ (1 + exp(- tr33));
Axz=inv(Hx'*Hx+0.1*eye(size(Hx',1)))*Hx'*Hz;
Hy =1 ./ (1 + exp(- tr22));
Ayz=inv(Hy'*Hy+0.1*eye(size(Hy',1)))*Hy'*Hz;

 x1=Hx*Axz;
 x2=Hy*Ayz;
t(k)=corr2(x1,x2);
end
[p1,p2]=max(t);
tr=tp{p2};
for p=1:5
tu(1+(p-1)*50:p*50,:)=tr11(1+(tr(p)-1)*50:tr(p)*50,:);
end

for i=1:5

    trx=tu(1+(i-1)*50:i*50,:);
    tryy=tr22(1+(i-1)*50:i*50,:);
    trxz=tr33(1+(i-1)*50:i*50,:);
        for j=1
    trx=mapminmax(trx,0,1);
    tryy=mapminmax(tryy,0,1);
     Hx =1 ./ (1 + exp(- trx));
Hz =1 ./ (1 + exp(- trxz));
Axz=inv(Hx'*Hx+0.9*eye(size(Hx',1)))*Hx'*Hz;
Hy =1 ./ (1 + exp(- tryy));
Ayz=inv(Hy'*Hy+0.9*eye(size(Hy',1)))*Hy'*Hz;
 x1=Hx*Axz;
 x2=Hy*Ayz;
 trx=x1;
 tryy=x2;
        end

xz(1+(i-1)*50:i*50,:)=x1;
yz(1+(i-1)*50:i*50,:)=x2;

end
%     xz=mapminmax(xz,0,1);
%     yz=mapminmax(yz,0,1);
%     to=randperm(50);
% for ii=1:5
%     xz(to+(ii-1)*50,:)=xz(1+(ii-1)*50:ii*50,:);
%     yz(to+(ii-1)*50,:)=yz(1+(ii-1)*50:ii*50,:);
% mz(1+(ii-1)*50:ii*50,:)=repmat(mean(tr33(1+(ii-1)*50:ii*50,:)),50,1);
% end
% for j=1:50
% %     for ij=1:5
% %         myz(1+(ij-1)*50:ij*50,:)=repmat(mean(yz(1+(ij-1)*50:ij*50,:)),50,1);
% %     end
%     xz=mapminmax(xz,0,1);
%     yz=mapminmax(yz,0,1);
%     tr33=mapminmax(tr33,0,1);
% xz =1 ./ (1 + exp(- xz));
% yz =1 ./ (1 + exp(- yz));
% tr33=1 ./ (1 + exp(- tr33));
% Azz=inv(tr33'*tr33+0.9*eye(size(tr33',1)))*tr33'*mz;
% xz=xz*Azz;
% yz=yz*Azz;
% tr33=tr33*Azz;
% end
x1=xz;
x2=yz;

for i=1:5
  xx1(i,:)=reshape(x1(1+(i-1)*50:i*50,:),1,1024*50);
  xx2(i,:)=reshape(x2(1+(i-1)*50:i*50,:),1,1024*50);
end

 
 
 % Amz=inv(Hz'*Hz+0.5*eye(size(Hz',1)))*Hz'*x1_m;
% x1=1 ./ (1 + exp(- x1));
% x2=1 ./ (1 + exp(- x2));
% xz1=x1*Amz;
% xz1(1:50,:)=[];
% yz1=x2*Amz;
% yz1(1:50,:)=[];
% zz1=Hz*Amz;
% zz1(1:50,:)=[];
% x11=mean(xz1);
% x22=mean(yz1);
% x33=mean(zz1);
% dxz(i,j)=sqrt(sum((x11-x33).^2));
% dyz(i,j)=sqrt(sum((x22-x33).^2));
%     end
%  end
end

% 
%  tr11=x1;
%  tr22=x2;
%     end
% end
% x1=mapminmax(x1,0,1);
% x2=mapminmax(x2,0,1);
% for i=1:5
%     x1_m(1+(i-1)*50:i*50,:)=repmat(mean(x1(1+(i-1)*50:i*50,:)),50,1);
%     x2_m(1+(i-1)*50:i*50,:)=repmat(mean(x2(1+(i-1)*50:i*50,:)),50,1);
% end
% Hx =1 ./ (1 + exp(- x1));
% Axz=inv(Hx'*Hx+0.5*eye(size(Hx',1)))*Hx'*x1_m;
% Hy =1 ./ (1 + exp(- x2));
% Ayz=inv(Hy'*Hy+0.5*eye(size(Hy',1)))*Hy'*x3_m;
%  x1=Hx*Axz;
%  x2=Hy*Axz;
% %  k1(i,j)=abs(corr2(x1,tr1));
% %  k2(i,j)=abs(corr2(x2,tr2));
% %  k3(i,j)=abs(corr2(x1,tr3));
% %  k4(i,j)=abs(corr2(x2,tr3));
% %  xz{j}=x1;
% %  yz{j}=x2;
% 
% 
% %     [k11,k22]=max(k1);
% %     xz1(1+(i-1)*50:i*50,:)=xz{k22};
% %     [k33,k44]=max(k2);
% %     yz1(1+(i-1)*50:i*50,:)=yz{k44};
% 
% 
% for i=1:5
%  xx1(i,:)=reshape(x1(1+(i-1)*50:i*50,:),1,1024*50);
% end
%  for i=1:5
%  xx2(i,:)=reshape(x2(1+(i-1)*50:i*50,:),1,1024*50);
%  end
 
