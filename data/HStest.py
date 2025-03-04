import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch


class HSTestData(data.Dataset):
    def __init__(self, image_dir, use_3D=False):
        test_data = sio.loadmat(image_dir)
        self.use_3Dconv = use_3D
        self.lrx8 = np.array(test_data['lrx8'][...], dtype=np.float32)
        self.lrx4 = np.array(test_data['lrx4'][...], dtype=np.float32)
        self.lrx2 = np.array(test_data['lrx2'][...], dtype=np.float32)
        self.hr = np.array(test_data['hr'][...], dtype=np.float32)

    def __getitem__(self, index):
        lrx8 = self.lrx8[index, :, :, :]
        lrx4 = self.lrx4[index, :, :, :]
        lrx2 = self.lrx2[index, :, :, :]
        hr = self.hr[index, :, :, :]

        if self.use_3Dconv:
            hr, lrx2, lrx4, lrx8 = hr[np.newaxis, :, :, :], lrx2[np.newaxis, :, :, :], lrx4[np.newaxis, :, :, :], lrx8[np.newaxis, :, :, :]
            hr = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)
            lrx2 = torch.from_numpy(lrx2.copy()).permute(0, 3, 1, 2)
            lrx4 = torch.from_numpy(lrx4.copy()).permute(0, 3, 1, 2)
            lrx8 = torch.from_numpy(lrx8.copy()).permute(0, 3, 1, 2)
        else:
            hr = torch.from_numpy(hr.copy()).permute(2, 0, 1)
            lrx2 = torch.from_numpy(lrx2.copy()).permute(2, 0, 1)
            lrx4 = torch.from_numpy(lrx4.copy()).permute(2, 0, 1)
            lrx8 = torch.from_numpy(lrx8.copy()).permute(2, 0, 1)
        #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
        #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
        #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
        return lrx8, lrx4, lrx2, hr

    def __len__(self):
        return self.hr.shape[0]
