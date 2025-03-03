import torch
import torch.nn as nn
from timm import create_model
from timm.layers import resample_abs_pos_embed
from mmengine.optim import OptimWrapper

class MultiUNI(nn.Module):
    def __init__(self, num_classes):
        super(MultiUNI, self).__init__()

        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        
        self.embed_dim = self.backbone.embed_dim
        self.cls_linear_heads = nn.ModuleList()
        for i in range(num_classes-1):  # 只判断 image 中含不含阳性 token
            self.cls_linear_heads.append(nn.Linear(self.embed_dim, 1))
            
        self.num_classes = num_classes

    @property
    def device(self):
        return next(self.parameters()).device

    def load_backbone(self, ckpt, frozen=True):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.backbone.load_state_dict(params_weight, strict=False))
        
        # update_params = ['cls_token']
        update_params = []
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                if name in update_params:
                    param.requires_grad = True

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        prev_grid_size = self.backbone.model.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=self.num_classes,
        )
        x = x.view(B, -1, C)
        to_cat = []
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x
    
    def calc_logits(self, x: torch.Tensor):
        '''
        Return:
            img_logits: (bs, 1)
            token_logits: (bs, num_tokens, 1)
        '''
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb = self.forward_features(x)
        cls_token = feature_emb[:,0,:]  # (bs, C)

        pred_img_logits = []
        for i in range(self.num_classes-1):
            pred_img_logits.append(self.cls_linear_heads[i](cls_token))  # [(bs, 1),]
        
        pred_img_logits = torch.cat(pred_img_logits, dim=-1)  # (bs, num_cls)
        return pred_img_logits

    def forward(self, data_batch, mode, optim_wrapper=None):        
            if mode == 'train':
                return self.train_step(data_batch, optim_wrapper)
            if mode == 'val':
                return self.val_step(data_batch)
        
    def train_step(self, databatch, optim_wrapper:OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        # img_logits: (bs, num_classes)
        img_logits = self.calc_logits(input_x.to(self.device))

        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = torch.zeros_like(img_logits, dtype=torch.float32)
        for i, token_labels in enumerate(databatch['token_labels']):
            label_list = list(set([tk[-1]-1 for tk in token_labels]))  # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
            binary_matrix[i, label_list] = 1

        loss = loss_fn(img_logits, binary_matrix)
        optim_wrapper.update_params(loss)
        return loss

    def val_step(self, databatch):
        input_x = databatch['images']
        img_logits = self.calc_logits(input_x.to(self.device))
        databatch['pos_probs'] = torch.sigmoid(img_logits)  # (bs, num_classes-1)
        return databatch

