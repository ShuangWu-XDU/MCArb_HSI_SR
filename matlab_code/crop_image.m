function [out] = crop_image(img, patch_size, stride, factor, file_name)

% Normalization for CAVE dataset
% img = double(img)./65535;
% Normalization for Pavia Center dataset
% img = double(img)./8000;
% Normalization for Chikusei dataset
img = double(img);

[H, W, C] = size(img);
p = patch_size;
pat_col_num = 1:stride:(H - p + 1);
pat_row_num = 1:stride:(W - p + 1);
total_num = length(pat_col_num) * length(pat_row_num);
index = 1;

% crop a single patch from whole image
for i=1:length(pat_col_num)
    for j = 1:length(pat_row_num)
        up = pat_col_num(i);
        down = up + p - 1;
        left = pat_row_num(j);
        right = left + p - 1;
        hr = img(up:down, left:right, :);
        lrx2 = single(imresize(hr, 0.5));
        lrx4 = single(imresize(hr, 0.25));
        lrx8 = single(imresize(hr, 0.125));
        hr = single(hr);
        %file_path = strcat('./dataset/Pavia/train/block_', 'Pavia', '_', num2str(index), '.mat');
%         file_path = strcat('./dataset/Chikusei/train/block_', 'Chikusei.', '_', num2str(index), '.mat');
        file_path = strcat('./dataset/HoustonU/train/block_', 'HoustonU.', '_', num2str(index), '.mat');
        save(file_path,'lrx8','lrx4','lrx2', 'hr','-v6');
        index = index + 1;
        
    end
    
end
out = total_num;
end

        