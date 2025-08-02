
import torch
import numpy as np

from SSD300VGG16.DBox import DBox
from SSD300VGG16.Decoder import Decoder
from SSD300VGG16.model import SSD300_VGG16



def test_DecoderDryrun():
    dbox = DBox()
    dbox_nxywhs = dbox.nxywhs

    imgs = torch.tensor(np.random.random((3, 3, 300, 300)), dtype=torch.float)

    model = SSD300_VGG16(10)

    locs, confs = model.forward(imgs)

    decoder = Decoder()
    ret = decoder.forward(dbox_nxywhs, locs)

    print(ret)

def test_Decoder():
    dbox = torch.tensor(
        [[2.0, 3.0, 2.0, 3.0], 
         [3.0, 2.0, 3.0, 2.0]], dtype=torch.float)
    locs = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], 
          [2.0, 2.0, 2.0, 2.0]], 
         [[3.0, 3.0, 3.0,3.0], 
          [4.0, 4.0, 4.0, 4.0]]], dtype=torch.float)
    
    decoder = Decoder()
    ret = decoder.forward(dbox, locs)
    print(ret)