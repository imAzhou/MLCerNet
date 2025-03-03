from timm import create_model
from mmpretrain import get_model

def get_backbone(backbone_type, img_size=224):
    '''
    backbone_type: ['vit', 'dinov2', 'uni']
    '''
    if backbone_type == 'uni':
        backbone = create_model(
            "vit_large_patch16_224", img_size=img_size, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        feature_dim = backbone.embed_dim
        num_patches = backbone.patch_embed.num_patches
    elif backbone_type in ['vit', 'dinov2']:
        backbone_model_name = {
            'vit': 'vit-large-p16_in21k-pre_3rdparty_in1k-384px',
            'dinov2': 'vit-large-p14_dinov2-pre_3rdparty'
        }
        patch_size = 16 if backbone_type == 'vit' else 14
        backbone = get_model(
            backbone_model_name[backbone_type], 
            pretrained=False, 
            backbone=dict(out_type='raw', 
                          with_cls_token=False)
        ).backbone
        feature_dim = backbone.embed_dims
        feat_size = img_size // patch_size
        num_patches = feat_size * feat_size
    return backbone,feature_dim,num_patches


if __name__ == '__main__':
    import torch

    backbone,embed_dim,num_patches = get_backbone('dinov2')
    mockdata = torch.randn((1,3,224,224))
    output = backbone(mockdata)
    print()