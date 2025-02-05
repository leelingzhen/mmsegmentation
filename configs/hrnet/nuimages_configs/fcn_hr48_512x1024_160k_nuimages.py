_base_ = './fcn_hr18_512x1024_160k_nuimages.py'
model = dict(
    pretrained='./pretrained_weights/hrnetv2_w48-d2186c55.pth',
    #pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
