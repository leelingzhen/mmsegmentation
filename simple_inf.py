from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = './configs/hrnet/fcn_hr18s_512x1024_40k_cityscapes.py'
checkpoint_file = '/home/leelingzhen/mmsegmentation/mmsegmentation/checkpoints/hrnet/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img_name = "face_forward.jpg"
img = f'/home/leelingzhen/Downloads/sample_dl_imgs/{img_name}'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# visualize the results in a new window
model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
model.show_result(img, result, out_file=f'/home/leelingzhen/Desktop/output_masks_{img_name}', opacity=0.5)

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   model.show_result(frame, result, wait_time=1)
