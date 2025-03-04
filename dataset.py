import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
rot90 = transforms.RandomRotation((90,90))
rot_90 = transforms.RandomRotation((-90,-90))
h_flip = transforms.RandomHorizontalFlip(p=1)
v_flip = transforms.RandomVerticalFlip(p=1)
def s_rot(img):
    return rot90(rot90(img))
def t_flip(img):
    return h_flip(rot_90(img))
def org(img):
    return img
def s_flip(img):
    return h_flip(rot90(img))
augdict = {0:org, 1:rot90, 2:rot_90, 3:h_flip, 4:v_flip, 5:s_rot, 6:t_flip, 7:s_flip}
def aug(img, mode):
    return augdict[mode](img).reshape(1, *img.shape)

class setMaker(Dataset):
    def __init__(self,data, size, stride, label) :
        c, h, w = data.shape
        col_num = np.arange(0 , h - size + 1, stride)
        if label == 'train':
            self.aug_len = augdict.__len__()
            col_num = col_num[int(len(col_num)*.1) + 1 : ]
            row_num = np.arange(0, w - size + 1, stride)
            num = len(col_num) * len(row_num)
            # print(col_num)
            
        elif label == 'val':
            self.aug_len = 1
            col_num = col_num[ : int(len(col_num)*.1) + 1]
            # print(col_num)
            row_num = np.arange(0, w - size + 1, stride)
            num = len(col_num) * len(row_num) 
                
        # self.LRx8 = torch.zeros(num, c, size//8, size//8)
        # self.LRx4 = torch.zeros(num, c, size//4, size//4)
        # self.LRx2 = torch.zeros(num, c, size//2, size//2)
        self.HR = torch.zeros(num, c, size, size)        
        count = 0
        for i in col_num:
            for j in row_num:
                self.HR[count] = data[:,i:i+size,j:j+size]
                count += 1
    def __getitem__(self, index):
        aug_num = index%self.aug_len
        hr =  aug(self.HR[index//self.aug_len], aug_num)               
        lrx2 = F.interpolate(hr, scale_factor=.5, mode='bicubic')
        lrx4 = F.interpolate(hr, scale_factor=.25, mode='bicubic')
        lrx8 = F.interpolate(hr, scale_factor=.125, mode='bicubic')

        return lrx8.squeeze(), lrx4.squeeze(), lrx2.squeeze(), hr.squeeze()
        
    def __len__(self):
        return self.HR.shape[0] * self.aug_len

# class testMaker(Dataset):
#     def __init__(self,data, dataset_name):
#         test_size = {'Chikusei':256, 'Pavia':128,'HoustonU':512}[dataset_name]
#         data = data[:,:test_size,:]
#         c, h, w = data.shape
#         test_num = w // test_size
#         self.HR = torch.zeros(test_num, c, test_size, test_size)        
#         count = 0
#         for i in range(test_num):
#             self.HR[count] = data[:,:,i*test_size:i*test_size+test_size]
#             count += 1
#     def __getitem__(self, index):
#         hr = self.HR[index]
#         hr = hr.reshape(1, *hr.shape)
#         # lrx2 = F.interpolate(hr, scale_factor=.5, mode='bicubic')
#         lrx4 = F.interpolate(hr, scale_factor=.25, mode='bicubic')
#         lrx8 = F.interpolate(hr, scale_factor=.125, mode='bicubic')
#         return lrx8.squeeze(), lrx4.squeeze(), hr.squeeze()#lrx2.squeeze(),
        
#     def __len__(self):
#         return self.HR.shape[0]
class testMaker(Dataset):
    def __init__(self,datatest, dataset_name):
        test_size = {'Chikusei':256, 'Pavia':128,'HoustonU':512}[dataset_name]        
        c, h, w = datatest.shape
        test_num = h // test_size
        self.HR = torch.zeros(test_num, c, test_size, test_size)        
        count = 0
        for i in range(test_num):
            self.HR[count] = datatest[:,i*test_size:i*test_size+test_size,:]
            count += 1
    def __getitem__(self, index):
        hr = self.HR[index]
        hr = hr.reshape(1, *hr.shape)
        # lrx2 = F.interpolate(hr, scale_factor=.5, mode='bicubic')
        lrx4 = F.interpolate(hr, scale_factor=.25, mode='bicubic')
        lrx8 = F.interpolate(hr, scale_factor=.125, mode='bicubic')
        return lrx8.squeeze(), lrx4.squeeze(), hr.squeeze()#lrx2.squeeze(),
        
    def __len__(self):
        return self.HR.shape[0]