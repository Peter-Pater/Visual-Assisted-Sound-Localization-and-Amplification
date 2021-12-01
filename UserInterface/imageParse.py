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

# PRA
from  scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra

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
        x = int(imageInfo[1])
        y = int(imageInfo[2])
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
            # print(original_width, original_height)
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

        # print(x, y)
        # print(metric_depth.shape)
        xy_depth = metric_depth[y][x] - 2

        fov = np.radians(54)
        near = (np.tan(fov/2) * 1.0)
        half_H = metric_depth.shape[0] // 2
        half_W = metric_depth.shape[1] // 2
        
        r = np.sqrt(half_H ** 2 + half_W ** 2) # consider r as the diag

        new_x = (half_W - x) / r * near * xy_depth
        new_y = (half_H - y) / r * near * xy_depth

        # position = np.array([new_x, new_y, xy_depth])
        target_loc = [xy_depth, new_x, new_y] # adjust according to the simulated space
        # plt.imshow(metric_depth)
        # plt.show()
        print("sound source location at:", target_loc)

        ### ### ### ### ### ### ### ### ### ### ### ###
        os.system(f"rm {imageInfo[0]}") # read complete, delete image
        
        # beamforming
        mic_center = np.asarray([0.25, 0, -0.25])
        mic_locs = np.c_[
            [0.2275, -0.0225, -0.25],
            [0.2725, -0.0225, -0.25],
            [0.2725, 0.0225, -0.25],
            [0.2275, 0.0225, -0.25]
        ] # just hardcode them... don't bother
        # read the sound file
        audio = wavfile.read("../UserInterface/data/audio.wav")
        sample_rate, data = audio[0], audio[1].T
        signals = [data[0], data[1], data[4], data[5]]
        result = beamformer(mic_locs, target_loc, signals, sample_rate)
        result = result / np.linalg.norm(result) * np.linalg.norm(data[2])
        wavfile.write("data/audio_processed.wav", sample_rate, result.astype("float32"))
        # return position

def beamformer(mic_locs, target_loc, signals, sample_rate):
    Lg_t = 0.1 # filter size in seconds
    Lg = np.ceil(Lg_t*sample_rate)
    
    fft_len = 512
    mics = pra.Beamformer(mic_locs, sample_rate, N=fft_len, Lg=Lg)
    source = pra.soundsource.SoundSource(target_loc)

    mics.rake_delay_and_sum_weights(source)
    mics.filters_from_weights()

    # process
    output = fftconvolve(mics.filters[0], signals[0])
    for i in range(1, len(signals)):
        output += fftconvolve(mics.filters[i], signals[i])
    output
    return output



def main():
    while True:
        try:
             imageDepth()
        except KeyboardInterrupt:
            os.system(f"rm data/audio_processed.wav") # program complete, delete sound file
            return
    

if __name__ == "__main__":
    main()
        