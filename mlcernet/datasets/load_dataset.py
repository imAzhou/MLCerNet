from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .cls_dataset import ClsDataset

def load_data(cfg):
    def custom_collate(batch):
        images = [item[0] for item in batch]
        image_labels = [item[1] for item in batch]
        token_labels = [item[2] for item in batch]

        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)

        return {
            'images': images_tensor,
            'image_labels': imglabels_tensor,
            'token_labels': token_labels
        }

    train_transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_dataset = ClsDataset(cfg.data_root, cfg.train_annojson, train_transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                            pin_memory=True,
                            batch_size=cfg.train_bs, 
                            sampler = train_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    val_transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_dataset = ClsDataset(cfg.data_root, cfg.val_annojson, val_transform)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            pin_memory=True,
                            batch_size=cfg.val_bs, 
                            sampler = val_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    
    return train_loader, val_loader
