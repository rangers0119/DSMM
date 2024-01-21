clc
clear all
close all

%%%%%鐩稿悓璐熻浇0涓嶅悓杞??300,480,600
p=1;
a = load('D:\desktop\映射\相同负载0\Normal\normal_speed_300_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\plane_gear_broken\plane_gear_broken_speed_300_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\ring_fault_left\ring_fault_left_speed_300_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c 
a = load('D:\desktop\映射\相同负载0\ring_fault_right\ring_fault_right_speed_300_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\sun_gear_broken\sun_gear_broken_speed_300_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\Normal\normal_speed_480_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\plane_gear_broken\plane_gear_broken_speed_480_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\ring_fault_left\ring_fault_left_speed_480_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\ring_fault_right\ring_fault_right_speed_480_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\sun_gear_broken\sun_gear_broken_speed_480_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\Normal\normal_speed_600_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\plane_gear_broken\plane_gear_broken_speed_600_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\ring_fault_left\ring_fault_left_speed_600_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\ring_fault_right\ring_fault_right_speed_600_load_0_1.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c
a = load('D:\desktop\映射\相同负载0\sun_gear_broken\sun_gear_broken_speed_600_load_0_2.mat');
b = a.Signal_0.y_values.values;
c = b(1:51200,2);
Y = c';
Y1(p,:)=Y;
fs(p)=a.Signal_0.x_values.increment;
p=p+1;
clear Y x a b c

x1=Y1;

p=6;
% for i=1:5
% [x]=preclass_2(Y1(1:5,:),Y1(6:10,:));
% x1(1:5,:)=x;
% [x]=preclass_2(Y1(11:15,:),Y1(6:10,:));
% x1(11:15,:)=x;
 [xx1,xx2]=preclass_2(Y1(11:15,:),Y1(6:10,:),Y1(1:5,:));
 x1(11:15,:)=xx1;
 x1(6:10,:)=xx2;
%   x1(11:15,:)=xx2;
% % p=p+1;
% % end
% p=11;
% for i=6:10
%     if i==6
%         j=13;
%     elseif i==7
%         j=14;
%     elseif i==8
%         j=15;
%     elseif i==9
%         j=11;
%     else
%         j=12;
%     end
% [x]=preclass_1(Y1(i,:),Y1(j,:));
% x1(p,:)=x;
% p=p+1;
% end
pp=1;
Y=x1(pp,:);
pp=pp+1;
save prnormal_speed_300_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prplane_gear_broken_speed_300_load_0_1 Y                               
Y=x1(pp,:);
pp=pp+1;
save prring_fault_left_speed_300_load_0_1 Y                                 
Y=x1(pp,:);
pp=pp+1;
save prring_fault_right_speed_300_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prsun_gear_broken_speed_300_load_0_1 Y                              
Y=x1(pp,:);
pp=pp+1;
save prnormal_speed_480_load_0_1 Y                                
Y=x1(pp,:);
pp=pp+1;
save prplane_gear_broken_speed_480_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prring_fault_left_speed_480_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prring_fault_right_speed_480_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prsun_gear_broken_speed_480_load_0_1 Y        
Y=x1(pp,:);
pp=pp+1;
save prnormal_speed_600_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prplane_gear_broken_speed_600_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prring_fault_left_speed_600_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prring_fault_right_speed_600_load_0_1 Y
Y=x1(pp,:);
pp=pp+1;
save prsun_gear_broken_speed_600_load_0_2 Y