import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class Adapt_3d(nn.Module):
    def __init__(self,   kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=False, num_experts=4):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(1,1,*kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

    def forward(self, x, n_scale):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / n_scale
        routing_weights = self.routing(scale).view(self.num_experts, 1, 1)
        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        
        fused_weight = fused_weight.view(1, 1, *self.kernel_size)

        # convolution
        out = F.conv3d(x.unsqueeze(1), fused_weight, stride=self.stride, padding=self.padding).squeeze()

        return out + x
class Adapt_3d_2(nn.Module):
    def __init__(self,   n_feats = 32, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=False, num_experts=4):
        super().__init__()
        self.n_feats= n_feats
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(n_feats,n_feats,*kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

    def forward(self, x, n_scale):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / n_scale
        routing_weights = self.routing(scale).view(self.num_experts, 1, 1)
        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        
        fused_weight = fused_weight.view(self.n_feats, self.n_feats, *self.kernel_size)
        # convolution
        out = F.conv3d(x, fused_weight, stride=self.stride, padding=self.padding)

        return out + x
    
class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super().__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, n_scale):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / n_scale
        routing_weights = self.routing(scale).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out
class SA_adapt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, n_scale):
        mask = self.mask(x)
        adapted = self.adapt(x, n_scale)

        return x + adapted * mask

def grid_sample(x, offset, scale): 
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros',align_corners=True)

    return output

class Meta_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)#embedding 1, 64, rh, rw
        
        ## offsets
        offset = self.offset(embedding)#offset 1, 2, rh, rw
        ## filters
        routing_weights = self.routing(embedding)#routing_weights 1, num_experts, rh, rw

        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0 # x out is b, c , rh,rw
    
class Meta_upsample_2(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels*2, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels*2, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)#embedding 1, 64, rh, rw
        
        ## offsets
        offset = self.offset(embedding)#offset 1, 2, rh, rw
        ## filters
        routing_weights = self.routing(embedding)#routing_weights 1, num_experts, rh, rw

        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale*w), self.channels*2, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale*w), self.channels, self.channels*2)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0
class bandAttn_Meta_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False, alpha = .5):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.alpha = alpha

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv3d(channels, channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        
        # routing head
        self.routing_1 = nn.Sequential(
            nn.Conv3d(channels, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv3d(64, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        self.routing_2 = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        )

    def forward(self, x, scale): # x in is bs,c,h,w
        bs, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input = torch.cat((            #input bs, c ,4, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0),
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).expand(1,c,-1,-1,-1).to(device)
        
        input = (input.permute(0,1,3,4,2)*
torch.cat((x.mean(dim=-1).mean(dim=-1,keepdim=True).mean(dim = 0) , torch.ones(c,3).to(device)), dim=-1).reshape(1,c,1,1,4).to(device)).permute(0,1,4,2,3)
        # print(input.shape) [bs, c, 4, rh, rw]
        
        
        embedding = self.body(input)#embedding 1, 64, rh, rw
        # (2) predict offsets and grid sample

        ## offsets
        offset = self.offset(embedding.squeeze())#offset 128, 2, rh, rw
        ## grid sample
        fea0 = torch.ones(bs,c,scale * h,scale * w).to(device)
        for i in range(c):
            fea0[:,i:i+1,:,:] = grid_sample(x[:,i:i+1,:,:],offset[i:i+1],scale)    ## bs * h * w * c * 1      
        

        # (3) predict filters and spatially varying filtering
        ## filters
        routing_weights = self.routing_2(self.routing_1(embedding).squeeze(0))#routing_weights 1, num_experts, rh, rw

        routing_weights = routing_weights.reshape(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.reshape(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.reshape(1, round(scale*h), round(scale*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.reshape(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.reshape(1, round(scale*h), round(scale*w), self.channels, self.channels//8)

        ## spatially varying filtering
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## bs * h * w * c * 1
        out = torch.matmul(weight_compress.expand([bs, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([bs, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0#*self.alpha*(1-self.alpha)
    
class bandAttn_Meta_upsample_new(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False, alpha = .5):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.alpha = alpha

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))
        

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale): # x in is bs,c,h,w
        bs, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        band_vec = F.interpolate(x.mean((0,2,3)).reshape(1,1,c),size=h * scale, mode='linear').squeeze(0)# 1, rh
        band_mat = band_vec.transpose(0,1)*band_vec
        band_mat = (band_mat - band_mat.min()) / (band_mat.max() - band_mat.min())
        
        input = torch.cat((            #input 1, 4, rh, rw
            band_mat.unsqueeze(0),
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        # print(input.shape)# [bs, c, 4, rh, rw]
        
        
        embedding = self.body(input)#embedding 1, 64, rh, rw
        # (2) predict offsets and grid sample

        ## offsets
        offset = self.offset(embedding)#offset 128, 2, rh, rw
        ## grid sample

        fea0 = grid_sample(x, offset, scale)  ## bs * h * w * c * 1      
        

        # (3) predict filters and spatially varying filtering
        ## filters
        routing_weights = self.routing(embedding)#routing_weights 1, num_experts, rh, rw
        
        routing_weights = routing_weights.reshape(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (rh*rw) * n

        weight_compress = self.weight_compress.reshape(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.reshape(1, round(scale*h), round(scale*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.reshape(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.reshape(1, round(scale*h), round(scale*w), self.channels, self.channels//8)

        ## spatially varying filtering
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## bs * h * w * c * 1
        out = torch.matmul(weight_compress.expand([bs, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([bs, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0
class bandAttn_Meta_upsample_new_2(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False, alpha = .5):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.alpha = alpha

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))
        

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale): # x in is bs,c,h,w
        bs, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        band_peak_map = (x.argmax(1).float()/c).to(device)
        band_peak_map = F.interpolate(band_peak_map.unsqueeze(1), size = round(scale*h), mode='bicubic')
        input = torch.cat((            
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0),
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0).to(device)
        input = (input.expand(bs, -1, -1, -1).permute(0, 2, 3, 1) * \
            torch.cat((torch.rand(bs, 1, round(scale*h), round(scale*h)), torch.ones(bs, 3, round(scale*h), round(scale*h))), dim=1).\
            permute(0, 2, 3, 1).to(device)).permute(0,3,1,2).to(device)#input bs, 4, rh, rw
        # return input
        
        
        embedding = self.body(input)#embedding bs, 64, rh, rw
        # (2) predict offsets and grid sample

        ## offsets
        offset = self.offset(embedding)#offset bs, 2, rh, rw
        ## grid sample
        # return offset
        fea0 = grid_sample(x, offset, scale)  ## bs * h * w * c * 1      

        # (3) predict filters and spatially varying filtering
        ## filters
        routing_weights = self.routing(embedding)#routing_weights bs, num_experts, rh, rw

        routing_weights = routing_weights.reshape(bs, self.num_experts, round(scale*h) * round(scale*w)).permute(0, 2,1)      # (rh*rw) * n

        weight_compress = self.weight_compress.reshape(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.reshape(bs, round(scale*h), round(scale*w), self.channels//8, self.channels)
        
        weight_expand = self.weight_expand.reshape(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.reshape(bs, round(scale*h), round(scale*w), self.channels, self.channels//8)

        ## spatially varying filtering
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## bs * h * w * c * 1
        out = torch.matmul(weight_compress, fea)
        out = torch.matmul(weight_expand, out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0
    
class Meta_upsample_3d(nn.Module):
    def __init__(self, channels, num_experts=4, kernel = 64, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.kernel = kernel
        # experts
        weight_expand = []
        for i in range(self.num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(self.kernel, 1, 5, 3, 3)))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))
        weight_compress = []
        for i in range(self.num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(1, self.kernel, 5, 3, 3)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        # two FC layers
        # routing head
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, 4),
            nn.Softmax(1)
        )
        # offset head
        self.offset = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        )
        

    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5        
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5
        
        
        coor_h = coor_h.permute(1, 0)
        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0).to(device)
        input_2 = torch.cat((torch.ones(round(scale), 1).to(device)/scale, coor_h[:round(scale)]), dim=1).to(device)
        

        # (2) predict filters and offsets
        # embedding = self.body(input)#embedding 1, 64, rh, rw
        
        ## offsets
        offset = self.offset(input)#offset 1, 2, rh, rw
        ## filters
        routing_weights = self.routing(input_2)#routing_weights 1, num_experts, rh, rw
        weight_expand = torch.matmul(routing_weights, \
                                     self.weight_expand.view(self.num_experts, -1)).sum(0).view(self.kernel, 1, 5, 3, 3)
        weight_compress = torch.matmul(routing_weights, \
                                       self.weight_compress.view(self.num_experts, -1)).sum(0).view(1, self.kernel, 5, 3, 3)      # (rh*rw) * n
        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(1)         ## b * h * w * c * 1
        ## spatially varying filtering
        out = F.conv3d(fea, weight_expand, stride=1, padding=(2,1,1))
        out = F.conv3d(out, weight_compress, stride=1, padding=(2,1,1)).squeeze()

        return out + fea0 # x out is b, c , rh,rw

class Meta_upsample_3d_2(nn.Module):
    def __init__(self, channels, num_experts=4, kernel = 32,bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.kernel = kernel
        # experts
        
        weight_expand = []
        for i in range(self.num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(kernel, 1, 5, 3, 3)))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))
        weight_compress = []
        for i in range(self.num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(1, kernel, 5, 3, 3)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))
        weight_2 = []
        for i in range(num_experts):
            weight_2.append(nn.Parameter(torch.Tensor(channels, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_2[i], a=math.sqrt(5))
        self.weight_2 = nn.Parameter(torch.stack(weight_2, 0))
        # two FC layers
        # routing head
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing_2 = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, 4),
            nn.Softmax(1)
        )
        # offset head
        
        

    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5        
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5
        
        
        coor_h = coor_h.permute(1, 0)
        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        input_2 = torch.cat((torch.ones(round(scale), 1).to(device)/scale, coor_h[:round(scale)]), dim=1).to(device)
        

        # (2) predict filters and offsets
        # embedding = self.body(input)#embedding 1, 64, rh, rw
        
        ## offsets
        embedding = self.body(input)
        offset = self.offset(embedding)#offset 1, 2, rh, rw
        routing_weights_2 = self.routing_2(embedding)
        ## filters
        routing_weights = self.routing(input_2)#routing_weights 1, num_experts, rh, rw
        weight_expand = torch.matmul(routing_weights, self.weight_expand.view(self.num_experts, -1)).sum(0).view(self.kernel, 1, 5, 3, 3)
        weight_compress = torch.matmul(routing_weights, self.weight_compress.view(self.num_experts, -1)).sum(0).view(1, self.kernel, 5, 3, 3)      # (rh*rw) * n
        
        routing_weights_2 = routing_weights_2.view(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (h*w) * n

        weight_2 = self.weight_2.view(self.num_experts, -1)
        weight_2 = torch.matmul(routing_weights_2, weight_2)
        weight_2 = weight_2.view(1, round(scale*h), round(scale*w), self.channels, self.channels)
        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(1)         ## b * h * w * c * 1
        ## spatially varying filtering
        out = F.conv3d(fea, weight_expand, stride=1, padding=(2,1,1))
        out = F.conv3d(out, weight_compress, stride=1, padding=(2,1,1))
        out += fea
        out = out.permute(0, 3, 4, 2, 1)            ## b * h * w * c * 1
        ## spatially varying filtering
        
        out = torch.matmul(weight_2.expand([b, -1, -1, -1, -1]), out).squeeze(-1).permute(0, 3, 1, 2)

        return out + fea0 # x out is b, c , rh,rw 
    
class Meta_upsample_com(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5        
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5
       
        coor_h = coor_h.permute(1, 0)
        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        embedding = self.body(input)
        offset = self.offset(embedding)#offset 1, 2, rh, rw
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        return fea0


class Meta_upsample_com_3d(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        kernel_size = (5, 3, 3)
        pad_size = (2,1,1)
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )

        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size = kernel_size, padding=pad_size),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size = kernel_size, padding=pad_size)
        )
        
        

    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5        
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5
        
        
        coor_h = coor_h.permute(1, 0)
        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
      
        ## offsets
        embedding = self.body(input)
        offset = self.offset(embedding)#offset 1, 2, rh, rw
        
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        
        return fea0+self.conv(fea0.unsqueeze(1)).squeeze()
class Meta_upsample_com_3d_2(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        kernel_size = (5, 3, 3)
        pad_size = (2,1,1)
        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size = kernel_size, padding=pad_size),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size = kernel_size, padding=pad_size)
        )
    def forward(self, x, scale): # x in is b,c,h,w
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)#embedding 1, 64, rh, rw
        
        ## offsets
        offset = self.offset(embedding)#offset 1, 2, rh, rw
        ## filters
        routing_weights = self.routing(embedding)#routing_weights 1, num_experts, rh, rw

        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1).permute(0, 3, 1, 2)+ fea0

        return out + self.conv(out.unsqueeze(1)).squeeze() 
class Meta_upsample_3d_3(nn.Module):
    def __init__(self, channels, num_experts=4, kernel = 64, bias=False):
        super().__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.kernel = kernel
        # experts
        weight_expand = []
        for i in range(self.num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(self.kernel, 4, 5, 3, 3)))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))
        weight_compress = []
        for i in range(self.num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(4, self.kernel, 5, 3, 3)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        # two FC layers
        # routing head
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, 4),
            nn.Softmax(1)
        )
        # offset head
        self.offset = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        )
    def forward(self, x, scale): # x in is b,c,h,w
        b, feat,c, h, w = x.size()
        device = x.device
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(device),
                   torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5        
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5
        
        
        coor_h = coor_h.permute(1, 0)
        input = torch.cat((            #input 1, 3, rh, rw
            torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0).to(device)
        input_2 = torch.cat((torch.ones(round(scale), 1).to(device)/scale, coor_h[:round(scale)]), dim=1).to(device)
        

        # (2) predict filters and offsets
        # embedding = self.body(input)#embedding 1, 64, rh, rw
        
        ## offsets
        offset = self.offset(input)#offset 1, 2, rh, rw
        ## filters
        routing_weights = self.routing(input_2)#routing_weights 1, num_experts, rh, rw
        weight_expand = torch.matmul(routing_weights, \
                                     self.weight_expand.view(self.num_experts, -1)).sum(0).view(self.kernel, 4, 5, 3, 3)
        weight_compress = torch.matmul(routing_weights, \
                                       self.weight_compress.view(self.num_experts, -1)).sum(0).view(4, self.kernel, 5, 3, 3)      # (rh*rw) * n
        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x.reshape(b*feat,c, h, w), offset, scale).reshape(b,feat,c, round(scale*w), round(scale*w))               ## b * h * w * c * 1
        fea = fea0        ## b * h * w * c * 1
        ## spatially varying filtering
        
        out = F.conv3d(fea, weight_compress, stride=1, padding=(2,1,1))
        out = F.conv3d(out, weight_expand, stride=1, padding=(2,1,1))

        return out + fea0 # x out is b, c , rh,rw