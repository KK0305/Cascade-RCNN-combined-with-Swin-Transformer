_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
albu_train_transforms = [ dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=180,
        interpolation=1,
        p=0.5)]
# dict(
#     type='RandomBrightnessContrast',
#     brightness_limit=[0.1, 0.3],
#     contrast_limit=[0.1, 0.3],
#     p=0.2),
# dict(
#     type='RandomBrightnessContrast',
#     brightness_limit=[0.1, 0.3],
#     contrast_limit=[0.1, 0.3],
#     p=0.2),
# dict(type='ChannelShuffle', p=0.1),
# dict(
#     type='OneOf',
#     transforms=[
#         dict(type='Blur', blur_limit=3, p=1.0),
#         dict(type='MedianBlur', blur_limit=3, p=1.0)
#     ],
#     p=0.1)]
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth')))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='GtBoxBasedCrop', crop_size=(640,416)),
    # dict(type='Resize', img_scale=[(1600, 1064), (800, 532)], keep_ratio=True),
    dict(type='Resize', img_scale=[(1280,832), (640,416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1280,832), (640,416)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py