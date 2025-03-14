%% This is a demo code to show how to generate training and testing samples from the harvard datasets %%
clear;clc;
%% Step 1: settings
img_test_size = 256;
img_train_size = 128;
H = 1040;
W = 1392;
bands = 31;

%% Step 2: generate the testing images
test_namelist = dir('Harvard_test\*.mat');
test_img_num = length(test_namelist);

test_stride = img_test_size*2;
test_pat_col_num = 1:test_stride:(H - img_test_size + 1);
test_pat_row_num = 1:test_stride:(W - img_test_size + 1);
test_num = test_img_num*length(test_pat_col_num)*length(test_pat_row_num);
hr = zeros(test_num,img_test_size,img_test_size,bands,'single');
lrx2 = zeros(test_num,img_test_size*.5,img_test_size*.5,bands,'single');
lrx4 = zeros(test_num,img_test_size*.25,img_test_size*.25,bands,'single');
lrx8 = zeros(test_num,img_test_size*.125,img_test_size*.125,bands,'single');
test_count = 1;
for i = 1:test_img_num
    load([test_namelist(i).folder,'\',test_namelist(i).name]);
    ref = ref/max(max(max(ref)));
    lbl_ref = lbl .* ref;
    for j = 1:length(test_pat_col_num)
        for k = 1:length(test_pat_row_num)
            up = test_pat_col_num(j);
            down = up + img_test_size - 1;
            left = test_pat_row_num(k);
            right = left + img_test_size - 1;
            lbl_test_pic = lbl(up:down, left:right, :);
            if ismember(0,lbl_test_pic) == 1
                continue
            end
            test_pic = lbl_ref(up:down, left:right, :);
            if mean(test_pic,"all") < .01
                continue
            end
            hr(test_count,:,:,:) = test_pic;
            lrx2(test_count,:,:,:) = single(imresize(test_pic, 0.5));
            lrx4(test_count,:,:,:) = single(imresize(test_pic, 0.25));
            lrx8(test_count,:,:,:) = single(imresize(test_pic, 0.125));
            test_count = test_count + 1;            
        end
    end
end
hr = hr(1:test_count-1,:,:,:);
lrx2 = lrx2(1:test_count-1,:,:,:);
lrx4 = lrx4(1:test_count-1,:,:,:);
lrx8 = lrx8(1:test_count-1,:,:,:);
size(hr)
save('.\dataset\Harvard\test\Harvard_test.mat','hr','lrx2','lrx4','lrx8');

%% Step 3: generate the training images
train_namelist = dir('Harvard_train\*.mat');
train_img_num = length(train_namelist);
train_stride = img_train_size;
train_pat_col_num = 1:train_stride:(H - img_train_size + 1);
train_pat_row_num = 1:train_stride:(W - img_train_size + 1);
train_num = train_img_num*length(train_pat_col_num)*length(train_pat_row_num);
train_count = 1;
for i = 1:train_img_num
    load([train_namelist(i).folder,'\',train_namelist(i).name]);
    ref = ref/max(max(max(ref)));
    lbl_ref = lbl .* ref;
    for j = 1:length(train_pat_col_num)
        for k = 1:length(train_pat_row_num)
            up = train_pat_col_num(j);
            down = up + img_train_size - 1;
            left = train_pat_row_num(k);
            right = left + img_train_size - 1;
            lbl_train_pic = lbl(up:down, left:right, :);
            if ismember(0,lbl_train_pic) == 1
                continue
            end
            train_pic = lbl_ref(up:down, left:right, :);
            if mean(train_pic,"all") < .01
                continue
            end
            hr = train_pic;
            lrx2 = single(imresize(train_pic, 0.5));
            lrx4 = single(imresize(train_pic, 0.25));
            lrx8 = single(imresize(train_pic, 0.125));
            file_path = strcat('.\dataset\Harvard\train\block_', 'Harvard', '_', num2str(train_count), '.mat');
            save(file_path,'lrx8','lrx4','lrx2', 'hr','-v6');
            train_count = train_count + 1; 
        end
    end
end

%% Step 4: Please manually remove 10% of the samples to the folder of evals