import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import cv2
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse

from utils import mkdir, bit_decomposition, calc_psnr, get_mgrid
from module import BitINR_parallel  
from dataloader import ImageFitting_bitplane

def main():
    parser = argparse.ArgumentParser(description='Train BitINR Model with configurable parameters.')
    parser.add_argument('--gpu', default='0', help='GPU id(s) to use, e.g. "0,1,2"')
    parser.add_argument('--tag', default='test', help='Tag for saving model and logs')
    
    parser.add_argument('--img_path', default='./load/roof.png', help='Path to the input image')
    parser.add_argument('--img_size', type=int, default=128, help='Output image size (square)')
    parser.add_argument('--num_bits', type=int, default=16, help='Number of bits for decomposition (ex: 8 or 16)')

    parser.add_argument('--save', action='store_true', help='If set, save result images during training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size in the model')
    parser.add_argument('--n_layer', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--steps', type=int, default=200000, help='Total number of training steps')
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloader')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    tag = args.tag

    save_path = os.path.join('./save', f"{tag}_ternary_weight")
    log_path = os.path.join(save_path, 'summaries')
    image_save_path = os.path.join(save_path, 'result_image')
    model_save_path = os.path.join(save_path, 'save_weight')
    
    for path in [log_path, model_save_path]:
        mkdir(path)
    if args.save:
        mkdir(image_save_path)

    writer = SummaryWriter(os.path.join(log_path, tag))
    
    bitbasis = (2 ** torch.arange(args.num_bits)).cuda()
    
    if args.num_bits == 8:
        img = cv2.imread(args.img_path)
    else:
        img = cv2.imread(args.img_path, -1)
    if img is None:
        raise FileNotFoundError(f"Image not found at {args.img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm_factor = 255.0 if args.num_bits == 8 else 65535.0

    img = img.astype('float32') / norm_factor
    itm = transforms.ToTensor()(img)

    
    crop = torchvision.transforms.CenterCrop(min(itm.shape[-1], itm.shape[-2]))
    resize = torchvision.transforms.Resize(args.img_size)
    itm = (resize(crop(itm.unsqueeze(0))).squeeze() * norm_factor).round().clamp(0, norm_factor) / norm_factor
    itm_ = bit_decomposition(itm, args.num_bits)

    grid = get_mgrid(args.img_size, 2)
    dataloader = DataLoader(
        ImageFitting_bitplane(grid, itm_, num_bits=args.num_bits),
        batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers
    )
    bitinr = BitINR_parallel(in_dim=2, hidden_dim=args.hidden_dim, out_dim=3, 
                             n_layer=args.n_layer, num_nets=args.num_bits)
    bitinr = bitinr.cuda()
    
    optimizer = torch.optim.Adam(lr=args.lr, params=bitinr.parameters())
    milestones = [args.steps * i // 10 for i in range(1, 10)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input.cuda().float().reshape(-1,1, 2)
    ground_truth = ground_truth.cuda().float().reshape(-1,args.num_bits, 3)
    
    lossfn = nn.BCELoss()
    sigm = nn.Sigmoid()
    berloss = nn.L1Loss()
    
    pbar = tqdm(range(args.steps))
    
    if n_gpus > 1:
        bitinr = nn.parallel.DataParallel(bitinr)
    
    bermax = 1e9
    psnr_max = 0
    
    for step in pbar:
        model_output = bitinr(model_input)
        model_output = sigm(model_output)
        loss = lossfn(model_output, ground_truth)
        errors = berloss(model_output.round(), ground_truth).detach().cpu().item()
        
        bit_inr_res_temp = (model_output.round().reshape(-1, args.num_bits, 3) * bitbasis[None, :, None]).sum(dim=1) / norm_factor
        gt_bits_out_temp = (ground_truth.reshape(-1, args.num_bits, 3) * bitbasis[None, :, None]).sum(dim=1) / norm_factor
        psnr = calc_psnr(bit_inr_res_temp, gt_bits_out_temp)
        
        writer.add_scalars('psnr', {'valid': psnr}, step)
        writer.add_scalars('ber', {'valid': errors}, step)
        writer.add_scalars('loss', {'train': loss.item()}, step)
        
        if errors < bermax:
            torch.save(bitinr.state_dict(), os.path.join(model_save_path, 'save_weight.pth'))
            psnr = calc_psnr(bit_inr_res_temp, gt_bits_out_temp)
            bermax = errors
            if psnr > psnr_max:
                psnr_max = psnr

            if bermax == 0:
                print(f"Zero error achieved at step {step}")
                iter_file = os.path.join(save_path, 'iter.txt')
                with open(iter_file, 'a') as f:
                    f.write(f"\n{step}")
                break
            if args.save:
                if args.num_bits == 8:
                    saving_image = (bit_inr_res_temp.detach().cpu().reshape(args.img_size, args.img_size, 3) * norm_factor).clamp(0,norm_factor).int().numpy().astype(np.uint8)
                else:
                    saving_image = (bit_inr_res_temp.detach().cpu().reshape(args.img_size, args.img_size, 3) * norm_factor).round().clamp(0,norm_factor).int().numpy().astype(np.uint16)
                saving_image = cv2.cvtColor(saving_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_save_path, 'result.png'), saving_image)

        pbar.set_description(f'BER {bermax:.4f} / PSNR: {psnr:.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

if __name__ == '__main__':
    main()
