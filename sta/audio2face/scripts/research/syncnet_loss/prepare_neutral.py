"""Prepare the fixed BGR photo with the official compatibility face crop."""
from common import *
import cv2,numpy as np,io,argparse
p=argparse.ArgumentParser();p.add_argument('png');a=p.parse_args();image=cv2.imread(a.png);assert image.shape==(224,224,3);photo=cv2.resize(image[16:176,32:192],(224,224));buffer=io.BytesIO();np.save(buffer,photo);freeze(OUT/'neutral-bgr.npy',buffer.getvalue())
