from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import os
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import  Visualizer
import random
from detectron2.engine import DefaultPredictor
import time
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from branch_point_detector import BranchPointDetector

class DetectronBranchNetwork:
    def __init__(self, weight_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

        cfg.MODEL.WEIGHTS = os.path.join(weight_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
        # cfg.DATASETS.TEST = ("pruningTest", )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # your number of classes + 1

        self.predictor = DefaultPredictor(cfg)

    def output_branch_targets(self, img, convert_to_bgr=False, output_diagnostic=True, vec_offset_draw=50):
        img_orig = img
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outputs = self.predictor(img)
        detector = BranchPointDetector.from_detectron_output(outputs)
        detector.run()
        bps = detector.branch_points
        diagnostic = None
        if output_diagnostic:
            diagnostic = detector.generate_output_image(img_orig, vec_offset_draw=vec_offset_draw)

        return bps, diagnostic


def get_imgs(root):
    rez = []
    for base_folder, _, files in os.walk(root):
        for file in files:
            if file.endswith('.png') and not '_' in file:
                rez.append(os.path.join(base_folder, file))

    return rez



if __name__ == '__main__':

    WEIGHTS_LOC = r'C:\Users\davijose\Documents\model_weights\model_final_new.pth'
    img_root = r'C:\Users\davijose\Pictures\TrainingData\RealData\NewLabelledData\RozaCloudyAfternoon'
    output_folder = r'C:\Users\davijose\Pictures\TrainingData\RealData\NewLabelledData\detectron_training\forcindy'

    net = DetectronBranchNetwork(WEIGHTS_LOC)
    img_paths = get_imgs(img_root)

    for j, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        bps, diagnostic = net.output_branch_targets(img, convert_to_bgr=False, output_diagnostic=True)

        diagnostic = cv2.cvtColor(diagnostic, cv2.COLOR_BGR2RGB)
        plt.imshow(diagnostic)
        plt.show()





        #
        #
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # colors = [
        #     (255, 0, 0),
        #     (0, 255, 0),
        #     (0, 0, 255),
        #     (255, 255, 0),
        #     (255, 0, 255),
        #     (0, 255, 255),
        # ]
        #
        # img_rgb_masked = img_rgb.copy()
        # trunk_class = 1
        # sb_class = 3
        #
        # counters = defaultdict(lambda: 0)
        #
        # for i in range(len(outputs['instances'])):
        #
        #     instance = outputs['instances'][i]
        #     pred_class = int(instance.pred_classes.cpu()[0].numpy())
        #     mask = instance.pred_masks[0].cpu().numpy()
        #     if pred_class in [trunk_class, sb_class]:
        #
        #         name = 'trunk' if pred_class == trunk_class else 'sidebranch'
        #         output_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        #         output_mask[mask] = 255
        #
        #         cur_i = counters[pred_class]
        #
        #         counters[pred_class] += 1
        #         Image.fromarray(output_mask).save(os.path.join(output_folder, f'{j}_{name}_{cur_i}.png'))
        #
        #
        #     img_rgb_masked[mask] = colors[pred_class]
        #
        # Image.fromarray(img_rgb).save(os.path.join(output_folder, f'{j}.png'))
        # Image.fromarray(img_rgb_masked).save(os.path.join(output_folder, f'{j}_masked.png'))
