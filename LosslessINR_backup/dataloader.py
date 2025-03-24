import torch
from torch.utils.data import Dataset

    
class ImageFitting_bitplane(Dataset):
    def __init__(self, coord,gt_bits,num_bits=16):
        super().__init__()
        """
        Args:
            coord (Tensor): A coordinate grid tensor representing the spatial positions of image pixels.
            gt_bits (Tensor): A tensor containing the ground truth bitplane data for the image.
                              Expected shape is [H, W,Bits, C)
            num_bits (int, optional): The number of bitplanes to use from the ground truth data (default is 16).
        Return:
            Bitplane Images (Tensor): (H*W,Bits,channels)
            Grid (Tensor): (H*W*Bits,3)-Main method or (H*W,2) - others (Ternary or valid)
        """

        bits=gt_bits.shape[-2]
        chan=gt_bits.shape[-1]
        self.pixels = gt_bits.reshape(-1, bits,chan)[:,-num_bits:,:]
        self.coords = coord

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels