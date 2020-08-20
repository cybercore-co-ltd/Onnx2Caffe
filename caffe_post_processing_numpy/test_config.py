import numpy as np
# nms config
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)

# Anchor config
anchor_generator=dict(
    type='AnchorGenerator',
    ratios=[1.0],
    octave_base_scale=8,
    scales_per_octave=1,
    strides=[8, 16, 32, 64])
    
# Norm config
norm_data_cfg = dict(
    mean=np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 
    std=np.array([58.395, 57.12 , 57.375], dtype=np.float32),
    to_rgb=True)