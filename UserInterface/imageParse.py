import sys
import os
import glob
import time
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

        ### ### ### ### ### ### ### ### ### ### ### ###
        os.system(f"rm {imageInfo[0]}") # read complete, delete image

def main():
    while True:
        try:
            imageDepth()
        except KeyboardInterrupt:
            return
    

if __name__ == "__main__":
    main()
        