"""
Script to center crop all images into the largest same dimensions.
"""

import argparse
import os

import torchvision

from PIL import Image

OUTPUT = "output"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path of the root directory containing the images")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"{args.path} does not exist.")
        quit()
    file_list = os.listdir(args.path)
    min_height = 1e10
    min_width = 1e10
    for file in file_list:
        try:
            with Image.open(os.path.join(args.path, file)) as image:
                width, height = image.size
                min_width = min(min_width, width)
                min_height = min(min_height, height)
        except:
            pass
    print(f"Cropping to height x width of ({min_height}, {min_width})")
    os.makedirs(os.path.join(args.path, OUTPUT), exist_ok=True)
    crop_transform = torchvision.transforms.CenterCrop((min_height, min_width))
    for file in file_list:
        try:
            with Image.open(os.path.join(args.path, file)) as image:
                cropped = crop_transform(image)
                cropped.save(os.path.join(args.path, OUTPUT, file))
        except:
            pass
        