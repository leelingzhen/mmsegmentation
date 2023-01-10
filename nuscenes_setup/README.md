# Nuscenes setup for mmsegmentation

1. Download nuscenes dataset to path `dataroot`  
2. Download all train/val/test splits into `dataroot`  


for training on mmsegmentation, custom datasets have to be structured as such:  

```
nuimages
|-- images
  |--train
    |-- image_1.jpg
    |-- image_2.jpg
    |-- image_3.jpg
    ...
  |--val
  |--test
|-- annotations
  |-- images
    |-- mask_1.png
    |-- mask_2.png
    |-- mask_3.png
    ...
```
for more information, refer to [mmseg tutorials](https://mmsegmentation.readthedocs.io/en/latest/tutorials/customize_datasets.html)  

This script will render masks from `dataroot` and will automatically structure the new dataset at path `out_path`  
Run this script for each split, etc: running the script each time will generate images and mask for each train/val/test split

## Sample Usage
```
python render_images.py --version 'v1.0-train' --dataroot '<dataroot>' --out_dir '<out_path>' --sample_limit 100000
```
