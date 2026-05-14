import torch
import torch.nn as nn
from mmengine.optim import OptimWrapper
from .classifier import MLCerClassifier
from .backbone import get_backbone

class MLCerNet(nn.Module):
    def __init__(self, num_classes, backbone_type, use_lora, img_size):
        super(MLCerNet, self).__init__()
        assert backbone_type in ['vit', 'dinov2', 'uni', 'smartccs']
        self.backbone,self.embed_dim,self.num_patches = get_backbone(backbone_type, img_size)
        self.backbone_type = backbone_type
        self.use_lora = use_lora
        self.num_classes = num_classes
        if use_lora:
            from peft import LoraConfig

            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules = ["qkv", "proj", "fc1", "fc2"], 
                lora_dropout=0.1,
                bias="none",
            )
        self.classifier = MLCerClassifier(num_classes, self.num_patches, self.embed_dim)


    @property
    def device(self):
        return next(self.parameters()).device

    def load_backbone(self, ckpt, frozen=True):
        if ckpt is not None:
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
            elif self.backbone_type == 'smartccs':
                teacher_state = params_weight.get('teacher', params_weight)
                new_state_dict = {}
                for key,value in teacher_state.items():
                    if 'backbone' in key:
                        key = '.'.join(key.split('.')[1:])
                    new_state_dict[key] = value
                print(self.backbone.load_state_dict(new_state_dict, strict=False))
            else:
                print(self.backbone.load_state_dict(params_weight, strict=False))
        else:
            print(f'Skip loading {self.backbone_type} backbone checkpoint.')
        
        if self.use_lora:
            from peft import get_peft_model

            self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        else:
            if frozen:
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False
    
    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        if self.use_lora:
            from peft import get_peft_model

            self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        
        print(self.load_state_dict(params_weight, strict=True))


    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type in ['uni']:
            output = self.backbone.forward_features(x)
            output = output[:,1:,:]  # (bs, num_tokens, C)
        elif self.backbone_type in ['vit', 'dinov2']:
            output = (self.backbone(x))[0]  # (bs,L,C)
        elif self.backbone_type == 'smartccs':
            output = self.backbone(x, is_training=True)['x_norm_patchtokens']
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
