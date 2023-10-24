_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', 
    '../_base_/default_runtime2.py', '../_base_/schedules/schedule_40k.py'
]

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
    classes=('T1C', 'T2'),
    palette=[[128, 0, 0], [0, 0, 128]]
)

train_dataloader = dict(
    batch_size=128,
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
        reduce_zero_label=True))
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
        reduce_zero_label=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator



crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))