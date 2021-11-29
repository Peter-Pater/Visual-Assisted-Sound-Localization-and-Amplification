from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import time
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from PIL import Image


def readImage():
    # search for png file
    os.chdir(".")
    pngs = glob.glob("*.png")
    assert len(pngs) == 1 or len(pngs) == 0 # either there is no file to process or exactly one file
    if len(pngs) == 1:
        imageName = pngs[0]
        print(f"loading {imageName}")
        x = imageName[:-4].split('-')[1] # mouse X
        y = imageName[:-4].split('-')[2] # mouse Y
        return imageName, x, y

def imageDepth():
    imageInfo = readImage()
    if imageInfo is not None:
        imageName = imageInfo[0] # file name
        x = imageInfo[1]
        y = imageInfo[2]
        print(f"Image captured with mouse at ({x}, {y})")
        read_start = time.time()
        while time.time() - read_start < 1:
            try:
                time.sleep(0.1) # give the image some time to load
                image = Image.open(imageName) # read the image
                print("read success")
                break
            except:
                print("image is still loading")
        ### call monodepth to process the image here ###

        model_name = "mono+stereo_1024x320"
        pred_metric_depth = True

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_path = os.path.join("models", model_name)
        # paths
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        with torch.no_grad():
            input_image = image.convert('RGB')
            original_width, original_height = input_image.size
            print(original_width, original_height)
            input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        
        metric_depth = np.array(np.squeeze(metric_depth))
        metric_depth = cv2.resize(metric_depth, [original_width, original_height], interpolation = cv2.INTER_AREA)

        xy_depth = metric_depth[x][y]

        fov = np.radians(54)
        near = (np.tan(fov/2) * 1.0)
        H = metric_depth.shape[0] // 2
        W = metric_depth.shape[1] // 2
        
        new_x = (x - W) / W * xy_depth
        new_y = (y - H) / H * xy_depth

        position = np.array([new_x, new_y, xy_depth])
        # plt.imshow(metric_depth)
        # plt.show()

        ### ### ### ### ### ### ### ### ### ### ### ###
        os.system(f"rm {imageInfo[0]}") # read complete, delete image
        return position

def main():
    while True:
        try:
            imageDepth()
        except KeyboardInterrupt:
            return
    

if __name__ == "__main__":
    main()
        