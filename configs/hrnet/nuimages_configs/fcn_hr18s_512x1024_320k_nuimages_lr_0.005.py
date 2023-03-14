_base_ = './fcn_hr18_512x1024_320k_nuimages_lr_0.005.py'
model = dict(
    pretrained='./pretrained_weights/hrnetv2_w18_small-b5a04e21.pth',
    #pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
