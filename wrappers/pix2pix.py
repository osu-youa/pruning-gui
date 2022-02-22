import os
import sys
import torch
import numpy as np
from PIL import Image
import pathlib

# CHANGE THIS TO YOUR PATH
model_path = os.path.join(os.path.expanduser('~'), 'install', 'pytorch-CycleGAN-and-pix2pix')
sys.path.append(model_path)

file_root = pathlib.Path(__file__).parent.resolve()


from data.base_dataset import get_transform
from torchvision.transforms import Resize
from util.util import tensor2im
import pickle
from options.test_options import TestOptions
from models import create_model

class Pix2PixGAN(object):

    def __init__(self, model_name, input_nc=3, output_nc=3, output_size=None, epoch='latest'):

        with open(os.path.join(file_root, 'default_options.pickle'), 'rb') as fh:
            opt = pickle.load(fh)

        opt.checkpoints_dir = os.path.join(model_path, 'checkpoints')
        opt.name = model_name
        opt.input_nc = input_nc
        opt.output_nc = output_nc
        opt.epoch = epoch
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model_tf = get_transform(opt)  # Takes in a PIL Image


        # For testing, but also to deal with overhead of loading model into GPU
        test_input = {'A': torch.rand(1,input_nc,256,256), 'A_paths': ''}
        self.model.set_input(test_input)
        self.model.test()

        self.gan_resize = Resize((256, 256), antialias=True)
        self.output_resize = None
        if output_size is not None:
            self.output_resize = Resize((output_size[1], output_size[0]), antialias=False)

    @staticmethod
    def process_input_numpy_array(array):
        # Takes in a W x H x C uint8 array
        # Returns a Tensor that can be fed into forward()

        tensor = torch.from_numpy(array).permute(2,0,1)
        tensor = (tensor / 255 - 0.5) * 2

        return tensor


    def forward(self, rgb_img):

        if isinstance(rgb_img, np.ndarray):
            img_tensor = self.gan_resize(self.process_input_numpy_array(rgb_img))
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


if __name__ == '__main__':
    # Examples showing how to load RGB and flow data
    import os

    gan = Pix2PixGAN('synthetic_flow_pix2pix', input_nc=6, output_nc=1, output_size=(640, 480), epoch='best')


    test_folder = r'C:\Users\davijose\Pictures\TrainingData\RealData\NewLabelledData\UFORozaVideos\video_136'
    rgb_path = os.path.join(test_folder, '8.png')
    flow_path = os.path.join(test_folder, '8_f_1.png')

    rgb = np.asanyarray(Image.open(rgb_path)).astype(np.uint8)
    flow = np.asanyarray(Image.open(flow_path).resize((rgb.shape[1], rgb.shape[0]))).astype(np.uint8)

    combined = np.dstack([rgb, flow])
    seg = gan.forward(combined)

    # Visualize image
    import matplotlib.pyplot as plt
    plt.imshow(seg)
    plt.show()

