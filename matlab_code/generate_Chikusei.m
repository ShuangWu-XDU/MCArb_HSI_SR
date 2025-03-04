%% This is a demo code to show how to generate training and testing samples from the HSI %%
clc
clear
close all

addpath('include');

%% Step 1: generate the training and testing images from the original HSI
load('D:\FILES_ \HSI_SR\Data\Chikusei_MATLAB\HyperspecVNIR_Chikusei_20140729.mat');%% Please down the Chikusei dataset (mat format) from https://www.sal.t.u-tokyo.ac.jp/hyperdata/
%% center crop this image to size 2304 x 2048
img = chikusei(107:2410,144:2191,:);
clear chikusei;
% normalization
img = img ./ max(max(max(img)));
img = single(img);
%% select first row as test images
[H, W, C] = size(img);
test_img_size = 512;
% test_pic_num = floor(W / test_img_size);
% mkdir test;
% for j = 1:2
% for i = 1:8
%     left = (i - 1) * 256 + 1;
%     right = left + 256 - 1;
%     test = img((j - 1) * 256 + 1:j*256,left:right,:);
%     save(strcat('./test/Chikusei_test_', int2str(i+8*(j-1)), '.mat'),'test');
%     
% end
% end

%% the rest left for training
mkdir ('train');
img = img((test_img_size+1):end,:,:);
save('./train/Chikusei_train.mat', 'img');

%% Step 2: generate the testing images used in mains.py
% generate_test_data;

%% Step 3: generate the training samples (patches) cropped from the training images
generate_train_data;

%% Step 4: Please manually remove 10% of the samples to the folder of evals