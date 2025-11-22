checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_l_20230227-cef260d4.pth'
crop_size = (
    256,
    256,
)
data_root = 'data/plantseg115'
dataset_type = 'PlantSeg115Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True, dist_cfg=dict(backend='nccl'))
ham_norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attention_kernel_paddings=[
            2,
            [
                0,
                3,
            ],
            [
                0,
                5,
            ],
            [
                0,
                10,
            ],
        ],
        attention_kernel_sizes=[
            5,
            [
                1,
                7,
            ],
            [
                1,
                11,
            ],
            [
                1,
                21,
            ],
        ],
        depths=[
            3,
            5,
            27,
            3,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=[
            64,
            128,
            320,
            512,
        ],
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_l_20230227-cef260d4.pth',
            type='Pretrained'),
        mlp_ratios=[
            8,
            8,
            4,
            4,
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        type='MSCAN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        ham_channels=512,
        in_channels=[
            128,
            320,
            512,
        ],
        in_index=[
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=2,
        type='LightHamHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = []
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=1000, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='PlantSeg_train.txt',
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        data_root='data/plantseg115',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=False,
        type='PlantSeg115Dataset'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='PackSegInputs'),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/segnext_mscan-l_test'
