import torch
from torch import nn
import math
from abc import ABCMeta, abstractmethod

class Classifier(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes, num_patches, embed_dim):
        '''
        num_classes: positive classes number + 1
        '''
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.feat_size = int(math.sqrt(num_patches))
        self.loss_fn = nn.BCEWithLogitsLoss()
        super().__init__()
        
    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def calc_logits(self, img_tokens: torch.Tensor):
        '''
        Args:
            img_tokens: (bs,num_token,C)
        Return:
            img_logits: (bs, n_cls)
        '''
        pass
    
    def get_batch_gt(positive_logits, databatch):
        '''
        Args:
            positive_logits: (bs, n_cls)
            databatch: batch of data
        Return:
            binary_matrix: (bs, n_cls)
        '''
        binary_matrix = torch.zeros_like(positive_logits, dtype=torch.float32)
        for i, token_labels in enumerate(databatch['token_labels']):
            # GT label in [1,5], pred label in [0,4]
            label_list = list(set([tk[-1] -1 for tk in token_labels]))
            binary_matrix[i, label_list] = 1
        return binary_matrix

    @abstractmethod
    def calc_loss(self, img_tokens, databatch):
        '''
        Args:
            img_tokens: (bs,num_token,C)
            databatch: batch of data
        Return:
            loss: int
        '''
        pass

    @abstractmethod
    def set_pred(self,img_tokens, databatch):
        '''
        Args:
            img_tokens: (bs,num_token,C)
            databatch: batch of data
        Return:
            databatch: batch of data
        '''
        pass

