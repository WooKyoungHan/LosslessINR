
import torch
import os



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_mgrid_bit(sidelen,bit_num=2, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1. 
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    bit = torch.linspace(-1, 1, steps=bit_num)
    tensors = (*tensors,bit)
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, (dim + 1))
    return mgrid

def bit_decomposition(image,bits=8):
    '''
    Generate bit planes from image 
    Input : Tensor, (C,H,W)
    Output : Tensor, (H,W,Bits,C)
    '''
    image = image * (2**bits - 1)
    res = []
    for i in range(bits):
        res.append(image%2)
        image = image//2
    return torch.stack(res,dim=0).permute(2,3,0,1)

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
