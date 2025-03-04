import torch
import torch.nn as nn
from meta_sr import Adapt_3d_2, Meta_upsample,Meta_upsample_3d_3
import torch.nn.functional as F

           
def _to_4d_tensor(x, depth_stride=None):
   """Converts a 5d tensor to 4d by stacking
   the batch and depth dimensions."""
   x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
   if depth_stride:
       x = x[::depth_stride]  # downsample feature maps along depth dimension
   depth = x.size()[0]
   x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
   x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
   x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
   x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
   return x, depth


def _to_5d_tensor(x, depth):
   """Converts a 4d tensor back to 5d by splitting
   the batch dimension to restore the depth dimension."""
   x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
   x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
   x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
   return x

class twoUint(nn.Module):
   def __init__(self, wn, n_feats):
       super().__init__()    	
       self.relu = nn.ReLU(inplace=True)
       
       self.conv1 = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))
       self.conv2 = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))   

   def forward(self, x):

       out = self.conv1(x)
       out = self.relu(out)
       out = self.conv2(out)
       out = torch.add(x, out)
       
       return out        
       
class E_HCM(nn.Module):
   def __init__(self, wn, n_feats, n_twoUint):
       super().__init__()
       self.relu = nn.ReLU(inplace=True)
       
       self.conv1 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)))
       self.conv2 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,3,1), stride=1, padding=(1,1,0)))     
       self.conv3 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,3), stride=1, padding=(1,0,1))) 
       
       twoD_body = [twoUint(wn, n_feats)for _ in range(n_twoUint)] 
              
       self.twoD_body = nn.Sequential(*twoD_body)
       self.SA_adapt = Adapt_3d_2(n_feats)
   def forward(self, x,n_scale):

       out = self.conv1(x)
       out = self.relu(out)
       out = torch.add(self.conv2(out), self.conv3(out))
       
       t = out      
       out, depth = _to_4d_tensor(out, depth_stride=1)                                   
   
       out = self.twoD_body(out)
       
       out = _to_5d_tensor(out, depth)          
       out = torch.add(out, t)       
       out = self.SA_adapt(out, n_scale)
       out = torch.add(out, x)
      
       return out         
                                                                                                                                                             
class ad_meta_ERCSR(nn.Module):
   def __init__(self, n_colors=102,n_feats=64,n_E_HCM=4, n_twoUint=2,upmode = 'Meta_upsample_3d_3', experts = 4):
       super().__init__()
       
       self.n_E_HCM = n_E_HCM
       self.upmode = upmode

              

       wn = lambda x: torch.nn.utils.weight_norm(x)
       self.relu = nn.ReLU(inplace=True)
               
       head = []
       head.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
       head.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))        
       self.head = nn.Sequential(*head)


       body = [E_HCM(wn, n_feats, n_twoUint)for _ in range(self.n_E_HCM)]
           

       self.body = nn.Sequential(*body)
       
       self.reduceD = wn(nn.Conv3d(n_feats*self.n_E_HCM, n_feats, kernel_size=(1,1,1), stride=1))                 
       self.gamma = nn.Parameter(torch.ones(self.n_E_HCM))               

       end = []
       end.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
       end.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))
       self.end = nn.Sequential(*end)        
                       
       tail = []
       tail.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
       tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))
       self.tail = nn.Sequential(*tail)
       self.up = {  'Meta_upsample':Meta_upsample,                     
                    'Meta_upsample_3d_3':Meta_upsample_3d_3,\
                    }[upmode](channels=n_colors, num_experts=experts)
                                                           
   def forward(self, x,n_scale):

        CSKC = F.interpolate(x, scale_factor=n_scale, mode='nearest')
        x = x.unsqueeze(1)
        x = self.head(x)             
        LSC = x

        H = []
        for i in range (self.n_E_HCM):
            x = self.body[i](x, n_scale)
            if i == 0:
                feo = x
            else:
                feo = torch.cat((feo,x))
            H.append(x*self.gamma[i])
              
        x = torch.cat(H, 1)  

        x = self.reduceD(x)                    
        x = self.end(x)
        x = torch.add(x, LSC)
             
        if self.upmode == 'Meta_upsample':                                                                                      
            x = self.tail(x)      
            x = x.squeeze(1)
            x = self.up(x, n_scale) 
        else:
            x = self.up(x, n_scale)                                                                           
            x = self.tail(x)      
            x = x.squeeze(1)
          
        x = torch.add(x, CSKC)
        return x,feo.reshape(feo.shape[0]*feo.shape[1], feo.shape[2],feo.shape[3],feo.shape[4])