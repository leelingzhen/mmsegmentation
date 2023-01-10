# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
        # dict(type='MMSegWandbHook', by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
        #      init_kwargs={'entity': "leelingzhen", # The entity used to log on Wandb
        #                   'project': "mmsegmentation", # Project name in WandB
        #                   }), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        # # MMSegWandbHook is mmseg implementation of WandbLoggerHook. ClearMLLoggerHook, DvcliveLoggerHook, MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook, SegmindLoggerHook are also supported based on MMCV implementation.
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
