import os
import sys
import torch
import numpy as np
from PIL import Image

model_path = os.path.join(os.path.expanduser('~'), 'install', 'pytorch-CycleGAN-and-pix2pix')
sys.path.append(model_path)

from data.base_dataset import get_transform
from torchvision.transforms import Resize
from util.util import tensor2im
import pickle
from options.test_options import TestOptions
from models import create_model

class Pix2PixGAN(object):

    def __init__(self, model_name, input_nc=3, output_nc=3, output_size=None):

        with open('default_options.pickle', 'rb') as fh:
            opt = pickle.load(fh)

        opt.checkpoints_dir = os.path.join(model_path, 'checkpoints')
        opt.name = model_name
        opt.input_nc = input_nc
        opt.output_nc = output_nc
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model_tf = get_transform(opt)  # Takes in a PIL Image

        # For testing, but also to deal with overhead of loading model into GPU
        test_input = {'A': torch.rand(1,input_nc,256,256), 'A_paths': ''}
        self.model.set_input(test_input)
        self.model.test()

        self.output_resize = None
        if output_size is not None:
            self.output_resize = Resize((output_size[1], output_size[0]), antialias=False)



    def forward(self, rgb_img):

        if isinstance(rgb_img, np.ndarray):
            rgb_img = Image.fromarray(rgb_img)
            img_tensor = self.model_tf(rgb_img)
        elif not isinstance(rgb_img, torch.Tensor):
            img_tensor = self.model_tf(rgb_img)
        else:
            img_tensor = rgb_img

        img_input = {'A': img_tensor.view(-1, *img_tensor.shape), 'A_paths': ''}
        self.model.set_input(img_input)
        self.model.test()

        out = self.model.get_current_visuals()['fake']
        if self.output_resize is not None:
            out = self.output_resize(out)
        return tensor2im(out)


