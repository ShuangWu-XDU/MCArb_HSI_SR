import argparse
import os
import copy
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import three_channel_show, compare_PSNR, compare_SAM
from pathlib import Path
from tqdm import tqdm
import os

from model.SSPSR import SSPSR
from model.SSPSR2 import SSPSR_o
from model.MCnet import MCnet
from model.ERCSR import ERCSR
from model.common import *
from data import HSTrainingData
from data import HSTestData
# loss
from loss import HybridLoss
# from loss import HyLapLoss
from metrics import  Metric

#python /home/Meta_HSI_SR/main_com_x8.py
 
 
def main():
    # parsers
    ag = argparse.ArgumentParser()
    ag.add_argument("--cuda", type=str, required=False,default='0', help="select GPU")
    ag.add_argument("--batch_size", type=int, default=16, help="batch size, default set to 64")
    ag.add_argument("--epochs", type=int, default=60, help="epochs, default set to 20")
    ag.add_argument("--n_scale", type=int, default=8, help="n_scale, default set to 2")
    ag.add_argument("--dataset_name", type=str, default="CAVE", help="dataset_name, default set to dataset_name")#Pavia Chikusei HoustonU Washington Kennedy PaviaU CAVE Harvard
    ag.add_argument("--model_title", type=str, default="SSPSR-b", help="model_title, default set to model_title")#SSPSR MCnet SSPSR-b ERCSR
    ag.add_argument("--seed", type=int, default=3407, help="start seed for model")
    ag.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    ag.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")    
    ag.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    args = ag.parse_args()
    print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))
    #Load datasets
    print('===> Loading datasets')
    colors = {'HoustonU':48, 'Pavia':102, 'Chikusei':128, 'Washington':191, 'CAVE':31, 'Harvard': 31}[args.dataset_name]    
    
    train_set = HSTrainingData(image_dir='datasets/'+args.dataset_name+'/train', augment=True)
    eval_set = HSTrainingData(image_dir='datasets/'+args.dataset_name+'/eval', augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_set = HSTestData(image_dir = 'datasets/'+args.dataset_name+'/test/'+args.dataset_name+'_test.mat')
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
    print('device is {}'.format(device))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    
    #Build model
    print('===> Building model')
    model_class = args.model_title+'x'+str(args.n_scale)#
    root = 'checkpoints/' +  args.model_title+ '/' + args.dataset_name + '/' + model_class
    path =root + '/'+ time.strftime("%Y%m%d-%H-%M", time.localtime())
    if not os.path.exists(path):
        os.makedirs(path)
        print('dir maked: '+path)
    OUT_DIR = Path(path)
    np.save(OUT_DIR.joinpath('args.npy'), vars(args))#np.load('args.npy', allow_pickle=True).item()
    writer = SummaryWriter(path+'/log')

    net = {'MCnet':MCnet, 'SSPSR-b':SSPSR, 'SSPSR':SSPSR_o,  'ERCSR':ERCSR}[args.model_title](
                n_scale = args.n_scale,
                n_colors = colors
            ).to(device).train()  
    
    # print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net).to(device).train()
    L1_loss = torch.nn.L1Loss()
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)    

    best_sorce = {
        'psnr'  : 0.0,
        'sam'   : 180.0,
        'val_psnr'  : 0.0,
        'val_sam'   : 180.0,
        'epoch' : 0,
    }
    for epoch in range(0, args.epochs):
        loss_l1 = Metric('L1_loss')
        loop = tqdm(enumerate(train_loader), total =len(train_loader),leave = True)#, ncols = 100
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        if epoch == 40:
            optimizer.param_groups[0]['lr'] = 1e-5
        for index, (lrx8, lrx4, lrx2, hr) in loop:#(lrx4, lms, hr)
            optimizer.zero_grad()
            lrx8 = lrx8.to(device)
            hr = hr.to(device)
            srx8 = net(lrx8)
            lossx8 = h_loss(srx8, hr)
            loss = lossx8
            loss_l1.update(loss)
            loss.backward()
            optimizer.step()
            loop.set_postfix(L1_loss_avg = loss_l1.avg.item(),
                             lr = optimizer.param_groups[0]['lr'],
                             lx8 = lossx8.item(),                            
                             )
        torch.save(copy.deepcopy(net.module.state_dict()),OUT_DIR.joinpath('net_last.pth'))
        net.eval()
        val_psnrx8 = []
        val_samx8 = []
        #eVal
        for index, (lrx8, lrx4,_, hr) in enumerate(val_loader):#index, (lrx4, lms, hr)
            lrx8 = lrx8.to(device)
            hr = hr.to(device)            
            with torch.no_grad():
                srx8 = net(lrx8).cpu() 
                hr = hr.cpu()
                val_psnrx8.append(compare_PSNR(srx8, hr))
                val_samx8.append(compare_SAM(srx8, hr))
                
        writer.add_scalar('val_psnrx8', np.mean(val_psnrx8), epoch)
        writer.add_scalar('val_samx8', np.mean(val_samx8), epoch)

        if best_sorce["val_psnr"] < np.mean(val_psnrx8):
            best_sorce['val_psnr'] = np.mean(val_psnrx8)
            torch.save(copy.deepcopy(net.module.state_dict()),OUT_DIR.joinpath('net_val.pth'))

        test_psnrx8 = []
        test_samx8 = []
        #eVal
        for index, (lrx8, lrx4,_, hr) in enumerate(test_loader):#index, (lrx4, lms, hr)
            lrx8 = lrx8.to(device)          
            hr = hr.to(device)            
            with torch.no_grad():
                srx8 = net(lrx8).cpu()                
                hr = hr.cpu()
                test_psnrx8.append(compare_PSNR(srx8, hr))
                test_samx8.append(compare_SAM(srx8, hr))

        writer.add_scalar('psnrx8', np.mean(test_psnrx8), epoch)
        writer.add_scalar('samx8', np.mean(test_samx8), epoch)

        print("psnrx8: {:.2f} samx8: {:.2f}".format(np.mean(test_psnrx8), np.mean(test_samx8)))
        if best_sorce["psnr"] < np.mean(test_psnrx8):
            best_sorce['psnr'] = np.mean(test_psnrx8)
            torch.save(copy.deepcopy(net.module.state_dict()),OUT_DIR.joinpath('net_test.pth'))#.module

if __name__ == "__main__":
    main()