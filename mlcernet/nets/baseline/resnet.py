import torch
import torch.nn as nn
from timm import create_model
from mmengine.optim import OptimWrapper

class MultiResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiResNet, self).__init__()
        self.backbone = create_model("resnet50", pretrained=False, num_classes=0)
        self.embed_dim = self.backbone.num_features
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
        
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    
    def calc_logits(self, x: torch.Tensor):
        '''
        Return:
            img_logits: (bs, num_classes)
        '''
        # feature_emb.shape: (bs, C)
        feature_emb = self.backbone(x)

        pred_img_logits = []
        for i in range(self.num_classes-1):
            pred_img_logits.append(self.cls_linear_heads[i](feature_emb))  # [(bs, 1),]
        
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
