_base_ = [
    '../_base_/default_runtime2.py',
    '../_base_/schedules/schedule_80k.py'
]
in_channels = 768
img_size = 224
# checkpoint = './pretrained/vit_large_p16_384_20220308-d4efb41d.pth'
out_indices = [5, 7, 11]
data_preprocessor = dict(  # 数据预处理的配置项，通常包括图像的归一化和增强
    type='SegDataPreProcessor',  # 数据预处理的类型
    mean=[0, 0, 0],  # 用于归一化输入图像的平均值
    std=[1, 1, 1],  # 用于归一化输入图像的标准差
    bgr_to_rgb=False,  # 是否将图像从 BGR 转为 RGB
    pad_val=0,  # 图像的填充值
    seg_pad_val=255,
    size=(224, 224)) 
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='checkpoints/vit_base.pth',
    backbone=dict(
        type='MAE',
        img_size=(224, 224),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[5, 7, 11]),
    decode_head=dict(
        type='ATMHead',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        num_classes=3,
        num_layers=3,
        num_heads=12,
        use_stages=len(out_indices),
        embed_dims=in_channels // 2,
        loss_decode=dict(
            type='ATMLoss', num_classes=3, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(224, 224), stride=(149, 149)),
)


# jax use different img norm cfg

img_scale = (224, 224)

train_pipeline = [
    dict(type='LoadImageFromNumpy'),
    dict(type='LoadAnnotations'),
    dict(
        type='SegRandomResizedCrop',
        size=224,
        scale=(1.0, 1.0),
        backend='cv2',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')]
test_pipeline = [
    dict(type='LoadImageFromNumpy'),
    dict(type='LoadAnnotations'),
    dict(
        type='SegRandomResizedCropVal',
        size=224,
        scale=(1.0, 1.0),
        backend='cv2',
        interpolation='bicubic'),
    dict(type='PackSegInputs')]

dataset_type="BaseSegDataset"
data_root="segdataset"
metainfo = dict(
    classes=("Background", 'T1C', 'T2'),
    palette=[[0, 128, 0], [128, 0, 0], [0, 0, 128]]
)

train_dataloader = dict(
    batch_size=96,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        img_suffix='.npy',
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline,
        metainfo=metainfo,
        reduce_zero_label=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        img_suffix='.npy',
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        reduce_zero_label=False))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator


optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.03,
                 )
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'ln': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 }))
#
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=60900, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200, max_keep_ckpts=3),
    param_scheduler=dict(type='ParamSchedulerHook'))

# randomness
resume = False

