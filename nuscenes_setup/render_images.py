# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import argparse
import gc
import os
import random
import shutil
from typing import List
from collections import defaultdict
from progress.bar import Bar

import cv2
import tqdm

import matplotlib.pyplot as plt
import numpy as np
from nuimages.nuimages import NuImages
from PIL import Image


def render_images(nuim: NuImages,
                  mode: str = 'all',
                  cam_name: str = None,
                  log_name: str = None,
                  sample_limit: int = 50,
                  filter_categories: List[str] = None,
                  out_type: str = 'image',
                  out_dir: str = '~/Downloads/nuImages',
                  cleanup: bool = True) -> None:
    """
    Render a random selection of images and save them to disk.
    Note: The images rendered here are keyframes only.
    :param nuim: NuImages instance.
    :param mode: What to render:
      "image" for the image without annotations,
      "annotated" for the image with annotations,
      "trajectory" for a rendering of the trajectory of the vehice,
      "all" to render all of the above separately.
    :param cam_name: Only render images from a particular camera, e.g. "CAM_BACK'.
    :param log_name: Only render images from a particular log, e.g. "n013-2018-09-04-13-30-50+0800".
    :param sample_limit: Maximum number of samples (images) to render. Note that the mini split only includes 50 images.
    :param filter_categories: Specify a list of object_ann category names. Every sample that is rendered must
        contain annotations of any of those categories.
    :param out_type: The output type as one of the following:
        'image': Renders a single image for the image keyframe of each sample.
        'video': Renders a video for all images/pcls in the clip associated with each sample.
    :param out_dir: Folder to render the images to.
    :param cleanup: Whether to delete images after rendering the video. Not relevant for out_type == 'image'.
    """
    # Check and convert inputs.
    assert out_type in ['image', 'video'], ' Error: Unknown out_type %s!' % out_type
    all_modes = ['image', 'annotated', 'trajectory']
    assert mode in all_modes + ['all'], 'Error: Unknown mode %s!' % mode
    assert not (out_type == 'video' and mode == 'trajectory'), 'Error: Cannot render "trajectory" for videos!'

    if mode == 'all':
        if out_type == 'image':
            modes = all_modes
        elif out_type == 'video':
            modes = [m for m in all_modes if m not in ['annotated', 'trajectory']]
        else:
            raise Exception('Error" Unknown mode %s!' % mode)
    else:
        modes = [mode]

    if filter_categories is not None:
        category_names = [c['name'] for c in nuim.category]
        for category_name in filter_categories:
            assert category_name in category_names, 'Error: Invalid object_ann category %s!' % category_name

    # Create output folder.
    out_dir = os.path.expanduser(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Filter by camera.
    sample_tokens = [s['token'] for s in nuim.sample]
    if cam_name is not None:
        sample_tokens_cam = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            key_camera_token = sample['key_camera_token']
            sensor = nuim.shortcut('sample_data', 'sensor', key_camera_token)
            if sensor['channel'] == cam_name:
                sample_tokens_cam.append(sample_token)
        sample_tokens = sample_tokens_cam

    # Filter by log.
    if log_name is not None:
        sample_tokens_cleaned = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            log = nuim.get('log', sample['log_token'])
            if log['logfile'] == log_name:
                sample_tokens_cleaned.append(sample_token)
        sample_tokens = sample_tokens_cleaned

    # Filter samples by category.
    if filter_categories is not None:
        # Get categories in each sample.
        sd_to_object_cat_names = defaultdict(lambda: set())
        for object_ann in nuim.object_ann:
            category = nuim.get('category', object_ann['category_token'])
            sd_to_object_cat_names[object_ann['sample_data_token']].add(category['name'])

        # Filter samples.
        sample_tokens_cleaned = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            key_camera_token = sample['key_camera_token']
            category_names = sd_to_object_cat_names[key_camera_token]
            if any([c in category_names for c in filter_categories]):
                sample_tokens_cleaned.append(sample_token)
        sample_tokens = sample_tokens_cleaned

    # Get a random selection of samples.
    random.shuffle(sample_tokens)

    # Limit number of samples.
    sample_tokens = sample_tokens[:sample_limit]

    print('Rendering %s for mode %s to folder %s...' % (out_type, mode, out_dir))
    for sample_token in tqdm.tqdm(sample_tokens):
        sample = nuim.get('sample', sample_token)
        log = nuim.get('log', sample['log_token'])
        log_name = log['logfile']
        key_camera_token = sample['key_camera_token']
        sensor = nuim.shortcut('sample_data', 'sensor', key_camera_token)
        sample_cam_name = sensor['channel']
        sd_tokens = nuim.get_sample_content(sample_token)

        # We cannot render a video if there are missing camera sample_datas.
        if len(sd_tokens) < 13 and out_type == 'video':
            print('Warning: Skipping video for sample token %s, as not all 13 frames exist!' % sample_token)
            continue

        for mode in modes:
            out_path_prefix = os.path.join(out_dir, '%s_%s_%s_%s' % (log_name, sample_token, sample_cam_name, mode))
            if out_type == 'image':
                write_image(nuim, key_camera_token, mode, '%s.jpg' % out_path_prefix)
            elif out_type == 'video':
                write_video(nuim, sd_tokens, mode, out_path_prefix, cleanup=cleanup)


def write_video(nuim: NuImages,
                sd_tokens: List[str],
                mode: str,
                out_path_prefix: str,
                cleanup: bool = True) -> None:
    """
    Render a video by combining all the images of type mode for each sample_data.
    :param nuim: NuImages instance.
    :param sd_tokens: All sample_data tokens in chronological order.
    :param mode: The mode - see render_images().
    :param out_path_prefix: The file prefix used for the images and video.
    :param cleanup: Whether to delete images after rendering the video.
    """
    # Loop through each frame to create the video.
    out_paths = []
    for i, sd_token in enumerate(sd_tokens):
        out_path = '%s_%d.jpg' % (out_path_prefix, i)
        out_paths.append(out_path)
        write_image(nuim, sd_token, mode, out_path)

    # Create video.
    first_im = cv2.imread(out_paths[0])
    freq = 2  # Display frequency (Hz).
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_path = '%s.avi' % out_path_prefix
    out = cv2.VideoWriter(video_path, fourcc, freq, first_im.shape[1::-1])

    # Load each image and add to the video.
    for out_path in out_paths:
        im = cv2.imread(out_path)
        out.write(im)

        # Delete temporary image if requested.
        if cleanup:
            os.remove(out_path)

    # Finalize video.
    out.release()

def convert_mask(input_arr: np.array) -> Image:
    translate_dict = {0:0 , 1:1 , 2:2, 3:2, 4:2, 5:2, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:3, 13:4, 14:5, 15:5, 16:5, 17:5, 18:5, 19:5, 20:5, 21:5, 22:5, 23:5, 24:1, 31: 0}
    palette = np.array([
    [1,1,1], # unknown
    [245, 254, 184], #driveable surface
    [95, 235, 52] , # humans
    [52, 107, 235], #moveable object
    [232, 13, 252], #static object
    [150, 68, 5], #vehicles
], dtype=np.uint8)
    seg_map = np.vectorize(translate_dict.get)(input_arr)
    seg_img = Image.fromarray(np.uint8(seg_map)).convert("P")
    seg_img.putpalette(palette)
    return seg_img

def write_image(nuim: NuImages, sd_token: str, mode: str, out_path: str) -> None:
    """
    Render a single image of type mode for the given sample_data.
    :param nuim: NuImages instance.
    :param sd_token: The sample_data token.
    :param mode: The mode - see render_images().
    :param out_path: The file to write the image to.
    """
    if mode == 'image':
        nuim.render_image(sd_token, annotation_type='none', out_path=out_path)

    elif mode == 'annotated':
        #nuim.render_image(sd_token, annotation_type='all', with_category=True, with_attributes=True, box_line_width=-1, out_path=out_path)
        semantic_mask, _ = np.array(nuim.get_segmentation(sd_token))
        semantic_mask = convert_mask(semantic_mask)
        semantic_mask.save(out_path.replace('.jpg','.png'))
        #mask_outpath = out_path + 'masked.jpg'

    elif mode == 'trajectory':
        # sample_data = nuim.get('sample_data', sd_token)
        # nuim.render_trajectory(sample_data['sample_token'], out_path=out_path)
        pass
    else:
        raise Exception('Error: Unknown mode %s!' % mode)

    # Trigger garbage collection to avoid memory overflow from the render functions.
    gc.collect()

def org_out_dir(
        data_dir: str,
        out_dir: str,
        split: int = 0.9
        ):
    out_dir = os.path.expanduser(out_dir)
    img_out_dir = os.path.join(out_dir, 'images')
    mask_out_dir = os.path.join(out_dir, 'annotations')
    if not os.path.isdir(img_out_dir) or not os.path.isdir(mask_out_dir):
        os.makedirs(img_out_dir)
        os.makedirs(os.path.join(img_out_dir, 'train'))
        os.makedirs(os.path.join(img_out_dir, 'val'))
        os.makedirs(mask_out_dir)
        os.makedirs(os.path.join(mask_out_dir, 'train'))
        os.makedirs(os.path.join(mask_out_dir, 'val'))

    data_dir = os.path.expanduser(data_dir)
    for file_name in tqdm.tqdm(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, file_name)):
            continue
        elif 'image' in file_name:
            os.rename(os.path.join(data_dir, file_name), os.path.join(img_out_dir, file_name))
        elif 'annotated' in file_name:
            os.rename(os.path.join(data_dir, file_name), os.path.join(mask_out_dir, file_name))

    split_point = int(len(os.listdir(img_out_dir)) * split)
    for i, file_name in tqdm.tqdm(enumerate(os.listdir(img_out_dir))):
        img_path = os.path.join(img_out_dir, file_name)

        # selected path is 'train' or 'val'
        if os.path.isdir(img_path):
            continue

        elif i <= split_point:
            os.rename(img_path, os.path.join(img_out_dir, 'train', file_name))
            annotated_file = file_name.rstrip('image.jpg') + 'annotated.png'
            os.rename(os.path.join(mask_out_dir, annotated_file), os.path.join(mask_out_dir, 'train', annotated_file))
        else:
            os.rename(img_path, os.path.join(img_out_dir, 'val', file_name))
            annotated_file = file_name.rstrip('image.jpg') + 'annotated.png'
            os.rename(os.path.join(mask_out_dir, annotated_file), os.path.join(mask_out_dir, 'val', annotated_file))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--seed', type=int, default=42)  # Set to 0 to disable.
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuimages')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--cam_name', type=str, default=None)
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--sample_limit', type=int, default=50)
    parser.add_argument('--filter_categories', action='append')
    parser.add_argument('--out_type', type=str, default='image')
    parser.add_argument('--out_dir', type=str, default='~/Downloads/nuImages')
    parser.add_argument('--split', type=float, default=0.9)
    args = parser.parse_args()

    # Set random seed for reproducible image selection.
    if args.seed != 0:
        random.seed(args.seed)

    # Initialize NuImages class.
    nuim_ = NuImages(version=args.version, dataroot=args.dataroot, verbose=bool(args.verbose), lazy=False)

    # Render images.
    render_images(nuim_, mode=args.mode, cam_name=args.cam_name, log_name=args.log_name, sample_limit=args.sample_limit,
                  filter_categories=args.filter_categories, out_type=args.out_type, out_dir=args.out_dir)
    # sort images
    org_out_dir(data_dir=args.out_dir, out_dir=args.out_dir, split=args.split)
