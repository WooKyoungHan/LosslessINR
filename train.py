import os
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse

from utils import mkdir, get_mgrid_bit, bit_decomposition, calc_psnr
from module import Siren
from dataloader import ImageFitting_bitplane#, ImageFitting_bitplane_16

def main():
    parser = argparse.ArgumentParser(
        description="Train LosslessINR model for bitplane image fitting (8-bit or 16-bit)."
    )
    parser.add_argument('--gpu', type=str, default='0', help='GPU id(s) to use (e.g., "0" or "0,1")')
    parser.add_argument('--tag', type=str, default='test', help='Experiment tag for saving logs and models')
    parser.add_argument('--save', action='store_true', help='If set, save result images during training')
    parser.add_argument('--flag', type=str, default='', help='Additional flagging tag for saving')

    parser.add_argument('--img_path', type=str, help='Path to the input image')
    parser.add_argument('--img_size', type=int, default=256, help='Size (width/height) for resized image')
    parser.add_argument('--num_bits', type=int, default=8, choices=[8, 16], help='Bit depth for decomposition (8 or 16)')
    
    parser.add_argument('--total_steps', type=int, default=10000, help='Total number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    
    save_path = os.path.join('./save', f"{args.tag}_{args.num_bits}bit",args.flag ,args.img_path.split('/')[-1].split('.')[-2])
    log_path = os.path.join(save_path, 'summaries')
    image_save_path = os.path.join(save_path, 'result_image')
    model_save_path = os.path.join(save_path, 'save_weight')
    
    for path in [log_path, model_save_path]:
        mkdir(path)
    if args.save:
        mkdir(image_save_path)

    writer = SummaryWriter(os.path.join(log_path, args.tag))
    
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
    itm = bit_decomposition(itm, args.num_bits)

    grid = get_mgrid_bit(args.img_size, args.num_bits, 2)
        
    dataset = ImageFitting_bitplane(grid, itm, num_bits=args.num_bits)

    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=8)
    
    model = Siren(
        in_features=3, out_features=3,
        hidden_features=512, hidden_layers=5,
        outermost_linear=True
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input.cuda().float().reshape(-1,args.num_bits, 3)
    ground_truth = ground_truth.cuda().float().reshape(-1,args.num_bits, 3)
    
    lossfn = nn.BCELoss()
    berloss = nn.L1Loss()
    sigm = nn.Sigmoid()
    
    pbar = tqdm(range(args.total_steps))
    
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
    
    bermax = 1e9
    psnr_max = 0 
    for step in pbar:
        model_output = model(model_input)
        model_output = sigm(model_output)
        
        loss = lossfn(model_output, ground_truth)
        errors = berloss(model_output.round(), ground_truth).detach().cpu().item()
        bit_inr_res_temp = (model_output.round().reshape(-1, args.num_bits, 3) * bitbasis[None, :, None]).sum(dim=1) / norm_factor
        gt_bits_out_temp = (ground_truth.reshape(-1, args.num_bits, 3) * bitbasis[None, :, None]).sum(dim=1) / norm_factor
        
        psnr = calc_psnr(bit_inr_res_temp, gt_bits_out_temp)

        

        
        if errors < bermax:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'save_weight.pth'))
            psnr = calc_psnr(bit_inr_res_temp, gt_bits_out_temp)
            bermax = errors
            if psnr > psnr_max:
                psnr_max = psnr

            if args.save:
                if args.num_bits == 8:
                    saving_image = (bit_inr_res_temp.detach().cpu().reshape(args.img_size, args.img_size, 3) * norm_factor).clamp(0,norm_factor).int().numpy().astype(np.uint8)
                else:
                    saving_image = (bit_inr_res_temp.detach().cpu().reshape(args.img_size, args.img_size, 3) * norm_factor).round().clamp(0,norm_factor).int().numpy().astype(np.uint16)
                saving_image = cv2.cvtColor(saving_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_save_path, 'result.png'), saving_image)

            if bermax == 0:
                print(f"Zero error achieved at step {step}")
                iter_file = os.path.join(save_path, 'iter.txt')
                with open(iter_file, 'a') as f:
                    f.write(f"\n{step}")
                break
                
        writer.add_scalars('psnr', {'valid': psnr}, step)
        writer.add_scalars('ber', {'valid': errors}, step)
        writer.add_scalars('loss', {'train': loss.item()}, step)

        pbar.set_description(f'BER {bermax:.4f} / PSNR: {psnr:.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
