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
from utils import  compare_PSNR, compare_SAM
from pathlib import Path
from tqdm import tqdm
import os
from SPP import SPP, SPP_3d
from model.ad_meta_MCnet import ad_meta_MCnet
from model.ad_meta_ERCSR import ad_meta_ERCSR
from model.ad_meta_SSPSR import ad_meta_SSPSR
from model.common import *
from data import HSTrainingData, HSTestData
# loss
from loss import HybridLoss, RLW, UncertaintyLoss
# from loss import HyLapLoss
from metrics import Metric

# global settings

# /home/ anaconda3/bin/python /home/ Meta_HSI_SR/main_x8.py --model_title SSPSR --upmode Meta_upsample_com_3d_2
 
# /home/ anaconda3/bin/python /home/ Meta_HSI_SR/main_x8.py --model_title MCnet --upmode Meta_upsample_3d_3
 
# /home/ anaconda3/bin/python /home/ Meta_HSI_SR/main_x8.py --model_title ERCSR --upmode Meta_upsample_3d_3
 
def main():
    # parsers
    ag = argparse.ArgumentParser()
    ag.add_argument("--experts", type=int, required=False,default=4, help="experts num")
    ag.add_argument("--upmode", required=False,default='Meta_upsample', help="upmode")#Meta_upsample_com_3d_2 Meta_upsample_3d_3
    ag.add_argument("--cuda", type=str, required=False,default='1', help="select GPU")
    ag.add_argument("--batch_size", type=int, default=4, help="batch size, default set to 64")
    ag.add_argument("--epochs", type=int, default=60, help="epochs, default set to 20")
    ag.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")#Chikusei Pavia HoustonU Washington Kennedy PaviaU
    ag.add_argument("--model_title", type=str, default="ERCSR", help="model_title, default set to model_title")#MCnet SSPSR GELIN ERCSR
    ag.add_argument("--seed", type=int, default=3407, help="start seed for model")
    ag.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    ag.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")    
    ag.add_argument("--gpus", type=str, default="0, 1", help="gpu ids (default: 7)")

    args = ag.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #Load datasets
    print('===> Loading datasets')

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
    colors = {'HoustonU':48, 'Pavia':102, 'Chikusei':128, 'Washington':191}[args.dataset_name]
    
    #Build model
    print('===> Building model')

    net = {'MCnet':ad_meta_MCnet, 'SSPSR':ad_meta_SSPSR, 'ERCSR':ad_meta_ERCSR}[args.model_title](#  ad_
        n_colors = colors, 
        upmode=args.upmode,
        experts=args.experts,
    ).to(device).train()

   
    if args.model_title == 'SSPSR':
        model_class = net.trunk.upsample._get_name() + '_adapt' if net._get_name()[:2] == 'ad' else \
                  net.trunk.upsample._get_name()#trunk.upsample up branch_up.up
    else:
        model_class = net.up._get_name() + '_adapt' if net._get_name()[:2] == 'ad' else \
                  net.up._get_name()#trunk.upsample up branch_up.up
    root = 'checkpoints/' +  args.model_title + '/'+ args.dataset_name + '/' + model_class +'_x8'
    path =root + '/'+ time.strftime("%Y%m%d-%H-%M", time.localtime())
    
    if not os.path.exists(path):
        os.makedirs(path)
        print('dir maked: '+path)
    OUT_DIR = Path(path)
    np.save(OUT_DIR.joinpath('args.npy'), vars(args))#np.load('args.npy', allow_pickle=True).item()
    writer = SummaryWriter(path+'/log')

    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)

    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_sorce = {
        'psnrx4'  : 0.0,
        'samx4'   : 180.0,
        'psnrx8'  : 0.0,
        'samx8'   : 180.0,
        'epoch' : 0,
    }
    for epoch in range(0, args.epochs):
        loss_l1 = Metric('L1_loss')
        loss_x8 = Metric('loss_x8')


        loop = tqdm(enumerate(train_loader), total =len(train_loader),leave = True)#, ncols = 100
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        if epoch == 40:
            optimizer.param_groups[0]['lr'] = 1e-5
        net.train()
        for index, (lrx8, lrx4, lrx2, hr) in loop:
            # if index == 1:
            #     break
                
            
            optimizer.zero_grad()
            lrx8 = lrx8.to(device)
            hr = hr.to(device)
            srx8 , feox8 = net(lrx8, n_scale = 8)#
            lossx8 = h_loss(srx8, hr)          
            loss = lossx8
            loss.backward()
            optimizer.step()
            loss_x8.update(lossx8)
            loss_l1.update(loss.sum())
            loop.set_postfix(
                             lr = optimizer.param_groups[0]['lr'],                            
                             x8 = loss_x8.avg.item(),)
        if epoch%5 == 0:
            torch.save(copy.deepcopy(net.state_dict()),OUT_DIR.joinpath('net_last.pth'))#.module
        net.eval()
        val_psnrx8 = []
        val_samx8 = []      
        for index, (lrx8, lrx4,_,hr) in enumerate(test_loader):
            lrx8 = lrx8.to(device)
            hr = hr.to(device)
            with torch.no_grad():
                srx8,_ = net(lrx8, n_scale = 8)
                srx8 = srx8.cpu()
                hr = hr.cpu()                
                val_psnrx8.append(compare_PSNR(srx8, hr))
                val_samx8.append(compare_SAM(srx8, hr))

        writer.add_scalar('psnrx8', np.mean(val_psnrx8), epoch)
        writer.add_scalar('samx8', np.mean(val_samx8), epoch)
        print("psnrx8: {:.2f} samx8: {:.2f}".format(np.mean(val_psnrx8), np.mean(val_samx8)))
        if best_sorce["psnrx8"] < np.mean(val_psnrx8):
            best_sorce['psnrx8'] = np.mean(val_psnrx8)
            torch.save(copy.deepcopy(net.state_dict()),OUT_DIR.joinpath('net_x8.pth'))#.module

if __name__ == "__main__":
    main()

    