import os
import numpy as np
import cv2
from tqdm import tqdm
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output')  # path to save as txt
parser.add_argument('--input', default="")  # path to save as txt
opt = parser.parse_args()

if not os.path.exists(opt.output):
    os.mkdir(opt.output)
if not os.path.exists(f"{opt.output}/images"):
    os.mkdir(f"{opt.output}/images")
if not os.path.exists(f"{opt.output}/edges"):
    os.mkdir(f"{opt.output}/edges")
if not os.path.exists(f"{opt.output}/vis"):
    os.mkdir(f"{opt.output}/vis")

name_list = []

prefix = opt.input.replace("/", "_")+"_"
for name in tqdm(os.listdir("./masks")):
    if not name.endswith(".png"):
        continue
    #os.system(f"cp masks/{name} {opt.output}/alpha/{name}")
    name_raw = name.replace(prefix, "")
    name_raw = "_".join(name_raw.split("_")[1:])
    name_raw = name_raw.replace(".png", ".jpg")
    os.system(f"cp images/{opt.input}/{name_raw} {opt.output}/images/{name}")
    vis = cv2.imread(f"images/{opt.input}/{name_raw}")
    msk = cv2.imread(f"masks/{name}", 0)
    msk0 = (msk>0).astype(np.uint8)*255
    ones = np.ones_like(msk)*255
    ones0 = np.ones_like(msk)*255
    ones[msk>0] = 0
    msk = np.stack([ones, ones0, ones], 2)
    vis = vis//2+msk//2
    cv2.imwrite(f"{opt.output}/vis/{name}", vis)
    cv2.imwrite(f"{opt.output}/edges/{name}", msk0)
    name_list.append(name+"\n")

with open(f"{opt.output}.txt", "w") as f:
    f.writelines(name_list)
