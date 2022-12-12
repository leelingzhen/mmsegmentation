import argparse
import gc
import os
import random
import splitfolders
import shutil
from progress.bar import Bar


def main(
        data_dir: str,
        out_dir: str = 'organised_nuImages',
        split: int = 0.9
        ):
    img_out_dir = os.path.join(os.getcwd(), out_dir, 'images')
    mask_out_dir = os.path.join(os.getcwd(), out_dir, 'annotations')
    if not os.path.isdir(img_out_dir) or not os.path.isdir(mask_out_dir):
        os.makedirs(img_out_dir)
        os.makedirs(os.path.join(img_out_dir, 'train'))
        os.makedirs(os.path.join(img_out_dir, 'val'))
        os.makedirs(mask_out_dir)
        os.makedirs(os.path.join(mask_out_dir, 'train'))
        os.makedirs(os.path.join(mask_out_dir, 'val'))

    data_dir = os.path.expanduser(data_dir)
    with Bar('moving to images/annotations...') as bar:
        for file_name in os.listdir(data_dir):
            if 'image' in file_name:
                os.rename(os.path.join(data_dir, file_name), os.path.join(img_out_dir, file_name))
            elif 'annotated' in file_name:
                os.rename(os.path.join(data_dir, file_name), os.path.join(mask_out_dir, file_name))
            bar.next()

    split_point = int(len(os.listdir(img_out_dir)) * split)
    with Bar('splitting to train/val...') as bar:
        for i, file_name in enumerate(os.listdir(img_out_dir)):
            if file_name == 'train' or file_name == "val":
                continue
            elif i <= split_point:
                os.rename(os.path.join(img_out_dir, file_name), os.path.join(img_out_dir, 'train', file_name))
                annotated_file = file_name.rstrip('image.jpg') + 'annotated.jpg'
                os.rename(os.path.join(mask_out_dir, annotated_file), os.path.join(mask_out_dir, 'train', annotated_file))
            else:
                os.rename(os.path.join(img_out_dir, file_name), os.path.join(img_out_dir, 'val', file_name))
                annotated_file = file_name.rstrip('image.jpg') + 'annotated.jpg'
                os.rename(os.path.join(mask_out_dir, annotated_file), os.path.join(mask_out_dir, 'val', annotated_file))
            bar.next()




if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='reorganise images')
    parser.add_argument('--out_dir', type=str, default='organised_nuImages')
    parser.add_argument('--data_dir', type=str, default='~/Downloads/nuImages')
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)

