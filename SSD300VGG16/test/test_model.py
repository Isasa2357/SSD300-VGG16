
import torch

import numpy as np

from SSD300VGG16.model import SSD300_VGG16

def test_dryrun():
    device = torch.device('cpu')
    model = SSD300_VGG16(21).to(device)

    imgs_4dryrun = torch.tensor(np.random.random((3, 3, 300, 300)), dtype=torch.float)

    ret = model.forward_printshape(imgs_4dryrun)