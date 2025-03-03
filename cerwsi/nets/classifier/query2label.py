import torch
import torch.nn as nn
import math
from types import SimpleNamespace
from .module import build_transformer,build_position_encoding
from .classifier import Classifier

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x
    
class Q2LClassifier(Classifier):
    def __init__(self, cfg):
        '''
        cfgs: common config
        '''
        super(Q2LClassifier, self).__init__(**cfg)

        args = {
            'feat_size': self.feat_size,
            'hidden_dim': 2048,
            'dim_feedforward': 8192,
            'enc_layers': 1,
            'dec_layers': 2,
            'nheads': 4,
            'position_embedding': 'sine',
            'dropout': 0.1,
            'pre_norm': False,
            'keep_other_self_attn_dec': False,
            'keep_first_self_attn_dec': False,
        }
        args = SimpleNamespace(**args)
       
        self.transformer = build_transformer(args)
        self.position_embedding = build_position_encoding(args)
        
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Conv2d(self.embed_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(self.num_classes-1, hidden_dim)
        self.fc = GroupWiseLinear(self.num_classes-1, hidden_dim, bias=True)
        

    def calc_logits(self, img_tokens: torch.Tensor):
        bs,num_tokens,C = img_tokens.shape
        inputx = img_tokens.reshape(bs, C, self.feat_size, self.feat_size)
        pos_emd = self.position_embedding(inputx).to(inputx.dtype)
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(inputx), query_input, pos_emd)[0] # B,K,d
        out = self.fc(hs[-1])  # (bs, num_classes-1)
        return out, None
    
    def calc_loss(self,img_tokens, databatch):
        positive_logits = self.calc_logits(img_tokens)
        binary_matrix = self.get_batch_gt(positive_logits, databatch)
        loss = self.loss_fn(positive_logits, binary_matrix)
        return loss

    def set_pred(self,img_tokens, databatch):
        positive_logits = self.calc_logits(img_tokens) # (bs, num_classes-1)
        databatch['pos_probs'] = torch.sigmoid(positive_logits) # (bs, num_classes-1)
        return databatch
