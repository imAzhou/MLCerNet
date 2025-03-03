
# strategy
lr = 0.0001
min_lr = 0.00001
weight_decay = 0.001
max_epochs = 50
warmup_epoch = 5
gamma = 0.95
save_each_epoch = False
frozen_backbone = True
use_lora = True
backbone_type = 'uni'
backbone_ckpt_config = {
    'vit': 'checkpoints/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth',
    'dinov2': 'checkpoints/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth',
    'uni': 'checkpoints/uni.bin',
}
backbone_ckpt = backbone_ckpt_config[backbone_type]
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay)
)

img_size = 224
