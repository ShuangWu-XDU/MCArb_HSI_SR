import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import utils


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_folders = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_files = self.image_folders
        # for i in self.image_folders:
        #     images = os.listdir(i)
        #     print(images)
        #     for j in images:
        #         if is_mat_file(j):
        #             full_path = os.path.join(i, j)
        #             self.image_files.append(full_path)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        lrx8 = np.array(data['lrx8'][...], dtype=np.float32)
        lrx4 = np.array(data['lrx4'][...], dtype=np.float32)
        lrx2 = np.array(data['lrx2'][...], dtype=np.float32)
        hr = np.array(data['hr'][...], dtype=np.float32)
        # ms = np.array(data['ms'][...], dtype=np.float32)
        # lms = np.array(data['ms_bicubic'][...], dtype=np.float32)
        # gt = np.array(data['gt'][...], dtype=np.float32)
        lrx8, lrx4, lrx2, hr = utils.data_augmentation(lrx8, mode=aug_num), utils.data_augmentation(lrx4, mode=aug_num), \
                        utils.data_augmentation(lrx2, mode=aug_num), utils.data_augmentation(hr, mode=aug_num)
        if self.use_3Dconv:
            # ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            lrx8, lrx4, lrx2, hr = lrx8[np.newaxis, :, :, :], lrx4[np.newaxis, :, :, :], lrx2[np.newaxis, :, :, :], hr[np.newaxis, :, :, :]
            lrx8 = torch.from_numpy(lrx8.copy()).permute(0, 3, 1, 2)
            lrx4 = torch.from_numpy(lrx4.copy()).permute(0, 3, 1, 2)
            lrx2 = torch.from_numpy(lrx2.copy()).permute(0, 3, 1, 2)
            hr = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)
        else:
            lrx8 = torch.from_numpy(lrx8.copy()).permute(2, 0, 1)
            lrx4 = torch.from_numpy(lrx4.copy()).permute(2, 0, 1)
            lrx2 = torch.from_numpy(lrx2.copy()).permute(2, 0, 1)
            hr = torch.from_numpy(hr.copy()).permute(2, 0, 1)
        return lrx8, lrx4, lrx2, hr

    def __len__(self):
        return len(self.image_files)*self.factor
