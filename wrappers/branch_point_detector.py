#!/usr/bin/env python3

# Read in masked images and estimate points where a side branch joins a leader (trunk)

import numpy as np
from matplotlib import image as MPLimage
from glob import glob
import csv
from os.path import exists
from os import listdir
import os
from collections import defaultdict
import cv2
from PIL import Image

class BranchPointDetector:

    @classmethod
    def from_detectron_output(cls, outputs):

        trunk_class = 1
        sb_class = 3

        images = []

        for i in range(len(outputs['instances'])):

            instance = outputs['instances'][i]
            pred_class = int(instance.pred_classes.cpu()[0].numpy())
            if pred_class in [trunk_class, sb_class]:

                mask = instance.pred_masks[0].cpu().numpy() * 255
                name = 'trunk' if pred_class == trunk_class else 'sidebranch'

                images.append({'name': name, 'image': mask.T})

        return cls(images)

    @classmethod
    def from_image_folder(cls, path, image_name):

        """ Read in all of the trunk/sidebranch/mask images, labeled by name
                @param path: Directory where files are located
                @param image_name: image number/name as a string
                @returns list of image, name, stats triplets"""

        images = []
        # search_path = f"{path}{image_name}_*.png"
        # fnames = glob(search_path)

        fnames = [f for f in os.listdir(path) if f.split('_')[0] == image_name and f.endswith('.png')]
        if not fnames:
            raise Exception('No images!')

        for filename in fnames:
            comps = filename.split('_')
            try:
                name = comps[1]
            except IndexError:
                name = 'original'

            im_data = {'name': name}
            im_orig = MPLimage.imread(os.path.join(path, filename))
            if len(im_orig.shape) is 2:
                im_data["image"] = np.transpose(im_orig)
            else:
                im_data["image"] = np.transpose(im_orig, axes=(1, 0, 2))

            images.append(im_data)

        return cls(images)


    def __init__(self, images):
        """ Detect possible branch points where side branch touches trunk
        @param path: Directory where files are located
        @param image_name: image number/name as a string"""

        self.images = images
        self.image_stats = {}
        self.classifications = defaultdict(list)
        self.branch_points = None

        for im in self.images:
            name = im['name']
            if name in ['trunk', 'sidebranch']:
                im['stats'] = self.stats_image(im['image'])
            self.classifications[name].append(im)

    def find_branch_points(self):
        branch_points = []
        for im_trunk in self.classifications['trunk']:
            for im_branch in self.classifications['sidebranch']:
                bp = self.find_branch_point(im_trunk, im_branch)
                if bp is not None:
                    branch_points.append(bp)

        return branch_points

    def run(self):
        self.branch_points = self.find_branch_points()

    def generate_output_image(self, base_img, output_path=None, vec_offset_draw=0):

        img = base_img.copy()
        for im in self.images:
            stats = im.get('stats')
            if stats is None:
                continue

            p1 = stats["lower_left"]
            p2 = stats["upper_right"]
            img = cv2.line(img, p1.astype(np.int64), p2.astype(np.int64), color=(128, 128, 128), thickness=3)

            pc = stats["center"]
            img = cv2.circle(img, pc.astype(np.int64), radius=5, color=(0, 128, 128), thickness=-1)

        for i, (p, v) in enumerate(self.branch_points):

            img = cv2.line(img, p.astype(np.int64), (p + v * 30).astype(np.int64), color=(128, 128, 128), thickness=3)
            text_loc = (p + vec_offset_draw * v).astype(np.int64)
            img = cv2.circle(img, text_loc, radius=5, color=(0,0,255), thickness=-1)
            img = cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
            img = cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if output_path:
            Image.fromarray(img).save(output_path)

        return img


    def stats_image(self, im):
        """ Add statistics (bounding box, left right, orientation, radius] to image
        @param im image
        @returns stats as a dictionary of values"""
        width = max(im.shape)
        height = min(im.shape)

        y_grid, x_grid = np.meshgrid(np.linspace(0.5, height - 0.5, height), np.linspace(0.5, width -  0.5, width))

        xs = x_grid[im > 0]
        ys = y_grid[im > 0]

        stats = {}
        stats["x_min"] = np.min(xs)
        stats["y_min"] = np.min(ys)
        stats["x_max"] = np.max(xs)
        stats["y_max"] = np.max(ys)
        stats["x_span"] = stats["x_max"] - stats["x_min"]
        stats["y_span"] = stats["y_max"] - stats["y_min"]

        avg_width = 0.0
        count_width = 0
        if stats["x_span"] > stats["y_span"]:
            stats["Direction"] = "left_right"
            stats["Length"] = stats["x_span"]
            for r in range(0, width):
                if sum(im[r, :]) > 0:
                    avg_width += sum(im[r, :] > 0)
                    count_width += 1
        else:
            stats["Direction"] = "up_down"
            stats["Length"] = stats["y_span"]
            for c in range(0, height):
                if sum(im[:, c]) > 0:
                    avg_width += sum(im[:, c] > 0)
                    count_width += 1
        stats["width"] = avg_width / count_width
        stats["center"] = np.array([np.mean(xs), np.mean(ys)])

        x_matrix = np.zeros([2, xs.shape[0]])
        x_matrix[0, :] = xs.transpose() - stats["center"][0]
        x_matrix[1, :] = ys.transpose() - stats["center"][1]
        covariance_matrix = np.cov(x_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        if eigen_values[0] < eigen_values[1]:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[1, :]
        else:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[0, :]
        eigen_ratio = stats["EigenValues"][1] / stats["EigenValues"][0]
        stats["EigenVector"][1] *= -1
        stats["EigenRatio"] = eigen_ratio
        stats["lower_left"] = stats["center"] - stats["EigenVector"] * (stats["Length"] * 0.5)
        stats["upper_right"] = stats["center"] + stats["EigenVector"] * (stats["Length"] * 0.5)
        print(stats)
        print(f"Eigen ratio {eigen_ratio}")
        return stats

    @staticmethod
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    @staticmethod
    def intersection(L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if abs(D) > 1e-10:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return None

    def find_branch_point(self, im_trunk, im_sidebranch):
        """ See if it makes sense to connect trunk to side branch
        @param im_trunk Trunk image and stats
        @param im_sidebranch Side branch image and stats
        @returns x,y location in image if connection, zero otherwise"""

        stats_trunk = im_trunk["stats"]
        stats_branch = im_sidebranch["stats"]

        if stats_trunk["EigenRatio"] < 50:
            print(f"Not a clean trunk {im_trunk['name']} {stats_trunk['EigenRatio']}")
            return None

        for end in ["lower_left", "upper_right"]:
            xy = stats_branch[end]

            l2 = np.sum((stats_trunk["upper_right"]-stats_trunk["lower_left"])**2)
            if abs(l2) < 0.0001:
                continue

            #The line extending the segment is parameterized as p1 + t (p2 - p1).
            #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

            #if you need the point to project on line extention connecting p1 and p2
            t = np.sum((xy - stats_trunk["lower_left"]) * (stats_trunk["upper_right"] - stats_trunk["lower_left"])) / l2

            #if you need to ignore if p3 does not project onto line segment
            if t > 1 or t < 0:
                print("   Not on trunk")
                continue

            l1 = BranchPointDetector.line(stats_trunk["lower_left"], stats_trunk["upper_right"])
            l2 = BranchPointDetector.line(stats_branch["lower_left"], stats_branch["upper_right"])
            pt_trunk = BranchPointDetector.intersection(l1, l2)
            pt_trunk_proj = stats_trunk["lower_left"] + t * (stats_trunk["upper_right"] - stats_trunk["lower_left"])
            if pt_trunk is None:
                pt_trunk = stats_trunk["lower_left"] + t * (stats_trunk["upper_right"] - stats_trunk["lower_left"])

            dist_to_trunk = np.sqrt(np.sum((pt_trunk - xy)**2))
            print(f"Trunk {im_trunk['name']} branch {im_sidebranch['name']} dist {dist_to_trunk}, {stats_trunk['width']}")
            vec_to_trunk = xy - pt_trunk
            if 0.25 * stats_trunk["width"] < dist_to_trunk < 1.75 * stats_trunk["width"]:
                if "lower_left" in end:
                    if vec_to_trunk[0] * stats_branch["EigenVector"][0] + vec_to_trunk[1] * stats_branch["EigenVector"][1] > 0:
                        print("  lower left")
                        pt_join = pt_trunk + stats_branch["EigenVector"] * stats_trunk["width"] * 0.5
                        return pt_join, stats_branch["EigenVector"]
                    else:
                        print("   Pointing wrong way")
                else:
                    if vec_to_trunk[0] * stats_branch["EigenVector"][0] + vec_to_trunk[1] * stats_branch["EigenVector"][1] < 0:
                        print("  upper right")
                        pt_join = pt_trunk + stats_branch["EigenVector"] * stats_trunk["width"] * -0.5
                        return pt_join, -stats_branch["EigenVector"]
                    else:
                        print("   Pointing wrong way")

        print("")
        return None


if __name__ == '__main__':
    # path = "./data/forcindy/"
    path = r'C:\Users\davijose\Pictures\TrainingData\RealData\NewLabelledData\detectron_training\forcindy'
    for im_i in range(0, 3):
        name = str(im_i)
        print(name)
        base_img = os.path.join(path, f'{name}_masked.png')
        bp = BranchPointDetector.from_image_folder(path, name)
        bp.run()
        bp.generate_output_image(np.array(Image.open(base_img)), os.path.join(path, f'{name}_annotated.png'))
