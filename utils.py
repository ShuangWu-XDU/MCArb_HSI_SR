import matplotlib.pyplot as plt
import numpy as np
import random
import torch
def three_channel_show(data,channel=(0,1,2),dataName='',figsize=(5,5),titlesize = 30):#input data shape is (channel,width,height)
    # data = np.array(data)
    if channel==(0,1,2):
        channel = tuple(random.sample(range(0, data.shape[0]), 3))
    x,y,z = channel
    sample_shape = data.shape
    c = torch.stack((data[x],data[y],data[z]))
    c = np.array(c)
    c = c.transpose(1,2,0)
    print('sample shape is '+str(sample_shape))
    print('sample channel is '+str(channel))    
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.title(dataName,fontdict={'weight':'normal','size': titlesize})
    plt.imshow(c)#imshow require (width,height,channel)

def compare_PSNR(img1, img2): # input torch: b, c, (h, w)
    mpsnr = 0
    for l in range(img1.size()[1]):

        mpsnr += 10. * torch.log10(1. / torch.mean((img1[:,l,:,:] - img2[:,l,:,:]) ** 2))

    return mpsnr / img1.size()[1]

def sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    num = 0
    sum_sam = 0
    x_true = np.array(x_true)
    x_pred = np.array(x_pred)
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    return (sum_sam / num) * 180 / np.pi
def compare_SAM(img1, img2):
    msam = []
    img1 = img1.permute(0,2,3,1)
    img2 = img2.permute(0,2,3,1)
    for i in range(img1.shape[0]):
        msam.append(sam(img2[i], img1[i]))
    return np.mean(msam)
def compare_RMSE(img1, img2):
    mse = 0
    for l in range(img1.size()[1]):

        mse += torch.sqrt(torch.mean((img1[:,l,:,:] - img2[:,l,:,:]) ** 2))

    return mse / img1.size()[1]
import scipy.io as sio
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader


# def load_dataset(dataset='train'):
#     num_channels = 3
#     if num_channels == 1:
#         is_gray = True
#     else:
#         is_gray = False

#     data_dir = './dataset'
#     set_name = ['bsds300']
#     if dataset == 'train':
#         print('Loading train datasets...')
#         train_set = get_training_set(data_dir, set_name, 128, 4, is_gray=is_gray)
#         return DataLoader(dataset=train_set, num_workers=8, batch_size=32,
#                           shuffle=True)
#     elif dataset == 'test':
#         print('Loading test datasets...')
#         test_set = get_test_set(data_dir, set_name, 4, is_gray=is_gray)
#         return DataLoader(dataset=test_set, num_workers=8, batch_size=16,
#                           shuffle=False)


def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))


# rescale every channel to between 0 and 1
def channel_scale(img):
    eps = 1e-5
    max_list = np.max((np.max(img, axis=0)), axis=0)
    min_list = np.min((np.min(img, axis=0)), axis=0)
    output = (img - min_list) / (max_list - min_list + eps)
    return output


# up sample before feeding into network
def upsample(img, ratio):
    [h, w, _] = img.shape
    return cv2.resize(img, (ratio*h, ratio*w), interpolation=cv2.INTER_CUBIC)


def bicubic_downsample(img, ratio):
    [h, w, _] = img.shape
    new_h, new_w = int(ratio * h), int(ratio * w)
    return cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)


def wald_downsample(data, ratio):
    [h, w, c] = data.shape
    out = []
    for i in range(c):
        dst = cv2.GaussianBlur(data[:, :, i], (7, 7), 0)
        dst = dst[0:h:ratio, 0:w:ratio, np.newaxis]
        out.append(dst)
    out = np.concatenate(out, axis=2)
    return out


def save_result(result_dir, out):
    out = out.numpy().transpose((0, 2, 3, 1))
    sio.savemat(result_dir, {'output': out})


def sam_loss(y, ref):
    (b, ch, h, w) = y.size()
    tmp1 = y.view(b, ch, h * w).transpose(1, 2)
    tmp2 = ref.view(b, ch, h * w)
    sam = torch.bmm(tmp1, tmp2)
    idx = torch.arange(0, h * w, out=torch.LongTensor())
    sam = sam[:, idx, idx].view(b, h, w)
    norm1 = torch.norm(y, 2, 1)
    norm2 = torch.norm(ref, 2, 1)
    sam = torch.div(sam, (norm1 * norm2))
    sam = torch.sum(sam) / (b * h * w)
    return sam


def extract_RGB(y):
    # take 4-2-1 band (R-G-B) for WV-3
    R = torch.unsqueeze(torch.mean(y[:, 4:8, :, :], 1), 1)
    G = torch.unsqueeze(torch.mean(y[:, 2:4, :, :], 1), 1)
    B = torch.unsqueeze(torch.mean(y[:, 0:2, :, :], 1), 1)
    y_RGB = torch.cat((R, G, B), 1)
    return y_RGB


def extract_edge(data):
    N = data.shape[0]
    out = np.zeros_like(data)
    for i in range(N):
        if len(data.shape) == 3:
            out[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            out[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return out


# def normalize_batch(batch):
#     # normalize using imagenet mean and std
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
#     return (batch - mean) / std


# def add_channel(rgb):
#     # initialize other channels using the average of RGB from VGG
#     R = torch.unsqueeze(y[:, 0, :, :], 1)
#     G = torch.unsqueeze(y[:, 1, :, :], 1)
#     B = torch.unsqueeze(y[:, 2, :, :], 1)
#     all_channel = torch.cat((B, B, G, G, R, R, R, R), 1)
#     return all_channel


# from LapSRN
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


