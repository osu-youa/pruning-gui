import numpy as np
from torchvision.transforms import Resize
import torch
import os



class ImageProcessor:

    def __init__(self, input_size, output_size, use_flow=False, gan_name=None, gan_input_channels=6, gan_output_channels=3):

        self.input_size = input_size        # Should be W x H
        self.output_size = output_size      # Should be W x H
        self.flownet_resize = None

        self.last_img = None
        self.last_flow = None

        self.flownet = None
        if use_flow:
            from wrappers.flownet import FlowNetWrapper
            self.flownet = FlowNetWrapper(cuda=True, weight_path=r'C:\Users\davijose\install\flownet2pytorch\models\FlowNet2_checkpoint.pth.tar')
            self.flownet_resize = Resize(tuple((np.array([input_size[1], input_size[0]]) // 64) * 64), antialias=True)

        self.gan = None
        self.gan_resize = None
        if gan_name is not None:
            from wrappers.pix2pix import Pix2PixGAN
            self.gan = Pix2PixGAN(gan_name, input_nc=gan_input_channels, output_nc=gan_output_channels, output_size=output_size)
            self.gan_resize = Resize((256, 256), antialias=True)


    def reset(self):
        self.last_img = None
        self.last_flow = None

    def process(self, img):

        # Image is assumed to be a H x W x C uint8 Numpy array
        img_tensor = torch.from_numpy(img).permute(2,0,1)

        if self.flownet is not None:
            # Scale the image to the closest dimensions possible divisible by 64
            img_tensor = self.flownet_resize(img_tensor).float()

            if self.last_img is None:
                self.last_img = img_tensor

            stacked_img = torch.stack([img_tensor, self.last_img]).transpose(1,0).float().cuda()
            stacked_img = stacked_img.reshape(1, *stacked_img.shape)
            rgb_flow = self.flownet.forward(stacked_img)

            self.last_flow = rgb_flow
            self.last_img = img_tensor

            img_tensor = torch.cat([img_tensor, torch.from_numpy(rgb_flow).permute(2,0,1)], dim=0)

        # # GAN accepts inputs as values between -1 and 1
        img_tensor = (img_tensor / 255 - 0.5) * 2

        if self.gan is not None:
            img_tensor = self.gan_resize(img_tensor)
            seg = self.gan.forward(img_tensor)
        else:
            raise NotImplementedError()

        # Convert back into Numpy image
        return seg





if __name__ == '__main__':

    processor = ImageProcessor((424,240), (128,128), use_flow=True, gan_name='orchard_cutterflowseg_pix2pix')
    from PIL import Image
    root = r'C:\Users\davijose\Pictures\TrainingData\GanTrainingPairsWithCutters\train'
    test_1 = os.path.join(root, 'render_6_randomized_00000.png')
    test_2 = os.path.join(root, 'render_6_randomized_00001.png')

    img_1 = np.array(Image.open(test_1)).astype(np.uint8)[:,:,:3]
    img_2 = np.array(Image.open(test_2)).astype(np.uint8)[:,:,:3]

    out_1 = processor.process(img_1)
    out_2 = processor.process(img_2)

    import matplotlib.pyplot as plt
    plt.imshow(out_2)
    plt.show()

    plt.imshow(processor.last_flow)
    plt.show()