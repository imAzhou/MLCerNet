import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from mmengine.optim import OptimWrapper
from .classifier import CerMClassifier
from .backbone import get_backbone

class CerMCNet(nn.Module):
    def __init__(self, num_classes, backbone_type, use_lora, img_size):
        super(CerMCNet, self).__init__()
        assert backbone_type in ['vit', 'dinov2', 'uni']
        self.backbone,self.embed_dim,self.num_patches = get_backbone(backbone_type, img_size)
        self.backbone_type = backbone_type
        self.use_lora = use_lora
        self.num_classes = num_classes
        if use_lora:
            self.lora_config = LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["qkv", "proj", "fc1", "fc2"],  # 应用 LoRA 的目标模块
                lora_dropout=0.1,  # Dropout 概率
                bias="none",  # 是否调整偏置
            )
        self.classifier = CerMClassifier(num_classes, self.num_patches, self.embed_dim)


    @property
    def device(self):
        return next(self.parameters()).device

    def load_backbone(self, ckpt, frozen=True):
        params_weight = torch.load(ckpt, map_location=self.device)
        if self.backbone_type in ['vit', 'dinov2']:
            new_state_dict = {}
            if self.backbone_type == 'vit':
                state_dict = params_weight
            if self.backbone_type == 'dinov2':
                state_dict = params_weight['state_dict']
            
            for key,value in state_dict.items():
                new_name = key.replace('backbone.', '')
                new_state_dict[new_name] = value
            print(self.backbone.load_state_dict(new_state_dict, strict=False))
        else:
            print(self.backbone.load_state_dict(params_weight, strict=False))
        
        if self.use_lora:
            self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        else:
            if frozen:
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False
    
    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        if self.use_lora:
            self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        
        print(self.load_state_dict(params_weight, strict=True))


    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type in ['uni']:
            output = self.backbone.forward_features(x)
            output = output[:,1:,:]  # (bs, num_tokens, C)
        elif self.backbone_type in ['vit', 'dinov2']:
            output = (self.backbone(x))[0]  # (bs,L,C)
        return output
    
    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        # img_logits: (bs, 1)
        # feature_emb.shape: (bs,img_token, C)
        feature_emb = self.extract_feature(input_x.to(self.device))
        loss = self.classifier.calc_loss(feature_emb, databatch)
        optim_wrapper.update_params(loss)
        return loss

    def val_step(self, databatch):
        input_x = databatch['images']
        feature_emb = self.extract_feature(input_x.to(self.device))
        databatch = self.classifier.set_pred(feature_emb, databatch)
        return databatch
