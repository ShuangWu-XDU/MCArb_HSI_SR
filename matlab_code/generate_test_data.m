fileFolder=fullfile('.\test\');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
% img_size = 128;
% bands = 102;
% img_size = 256;
% bands = 128;
img_size = 128;
bands = 48;

hr = zeros(numel(fileNames),img_size,img_size,bands);
lrx2 = zeros(numel(fileNames),img_size*.5,img_size*.5,bands);
lrx4 = zeros(numel(fileNames),img_size*.25,img_size*.25,bands);
lrx8 = zeros(numel(fileNames),img_size*.125,img_size*.125,bands);
cd test;
for i = 1:numel(fileNames)
    load(fileNames{i},'test');    
    hr(i,:,:,:) = test;
    lrx2(i,:,:,:) = single(imresize(test, 0.5));
    lrx4(i,:,:,:) = single(imresize(test, 0.25));
    lrx8(i,:,:,:) = single(imresize(test, 0.125));
end
cd ..;
hr = single(hr);
lrx2 = single(lrx2);
lrx4 = single(lrx4);
lrx8 = single(lrx8);
%save('.\dataset\Pavia\test\Pavia_test.mat','hr','lrx2','lrx4','lrx8');
% save('.\dataset\Chikusei\test\Chikusei_test.mat','hr','lrx2','lrx4','lrx8');
save('.\dataset\HoustonU\test\HoustonU_test.mat','hr','lrx2','lrx4','lrx8');