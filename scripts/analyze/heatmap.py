import argparse
from cerwsi.nets import CerMCNet
from cerwsi.utils import set_seed
import torch
import os
from pycocotools.coco import COCO
from PIL import ImageDraw
import json
from PIL import Image
from torchvision import transforms
from mmengine.config import Config
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

def draw_heatmap(attn_map, image, pred_result, item_inside, scale_factor, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    sq_x1,sq_y1,sq_w,sq_h = [0,0,406,406]
    draw = ImageDraw.Draw(image)
    for box_item in item_inside:
        category = box_item['clsname']
        x1, y1, x2, y2 = box_item['box_x1y1x2y2']
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1+sq_w, x2) - sq_x1
        y_max = min(sq_y1+sq_h, y2) - sq_y1
        
        color = 'red'
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 2, y_min - 15), category, fill=color)
    
    # 遍历每个类别并绘制其注意力图
    min_v,max_v = np.min(attn_map),np.max(attn_map)
    for i in range(attn_map.shape[0]):
        ax = axes[i // 3, i % 3]  # 计算对应的行和列
        ax.imshow(image, aspect='auto')  # 显示原图
        # expanded_map = zoom(attn_map[i], (scale_factor, scale_factor), order=0)  # 最近邻插值
        # ax.imshow(expanded_map, cmap='gray', alpha=0.5, vmin=0, vmax=1)  # 使用灰度图显示
        expanded_map = zoom(attn_map[i], (scale_factor, scale_factor), order=1)  # 双线性插值
        ax.imshow(expanded_map, cmap='hot', alpha=0.6, vmin=min_v, vmax=max_v)  # 使用热力图显示
        # ax.imshow(expanded_map, cmap='hot', alpha=0.6)  # 使用热力图显示
        ax.axis('off')  # 关闭坐标轴
        ax.set_title(classnames[i] + f': {pred_result[i]}')  # 设置标题，可以自定义
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def draw_heatmap_bs(bs_attn_map, bs_image_paths, bs_gt_labels, bs_pred_labels, save_path, clsid):
    min_v,max_v = torch.min(bs_attn_map),torch.max(bs_attn_map)
    bs = len(bs_image_paths)
    grid_size = 224
    axis_size = math.ceil(math.sqrt(bs))
    fig, axes = plt.subplots(axis_size, axis_size, figsize=(axis_size*4, axis_size*4))

    for bidx, attn_map, imgpath in zip(range(bs), bs_attn_map, bs_image_paths):
        image = Image.open(imgpath)
        purename = os.path.basename(imgpath).split('.')[0]
        feat_size = int(math.sqrt(len(attn_map)))
        attn_map_2d = attn_map.reshape(feat_size, feat_size).detach().cpu().numpy()
        scale_factor = grid_size // attn_map_2d.shape[-1]
        ax = axes[bidx // axis_size, bidx % axis_size]  # 计算对应的行和列

        image_resized = image.resize((grid_size, grid_size))
        ax.imshow(image_resized)  # 显示原图

        expanded_map = zoom(attn_map_2d, (scale_factor, scale_factor), order=1)  # 双线性插值
        ax.imshow(expanded_map, cmap='hot', alpha=0.6, vmin=min_v, vmax=max_v)  # 使用热力图显示
        ax.axis('off')  # 关闭坐标轴
        gt = ','.join(map(str, bs_gt_labels[bidx]))
        pred = ','.join(map(str, bs_pred_labels[bidx]))
        color = 'red' if clsid in bs_pred_labels[bidx] else 'black'
        ax.set_title(f'{purename}: {gt}/{pred}', color = color)  # 设置标题，可以自定义
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def draw_heatmap_clean(bs_attn_map, bs_image_paths):
    min_v,max_v = torch.min(bs_attn_map),torch.max(bs_attn_map)
    bs = len(bs_image_paths)
    grid_size = 224
    axis_size = math.ceil(math.sqrt(bs))
    fig, axes = plt.subplots(5, 5, figsize=(5*4, 5*4))

    # 调整子图间距
    # plt.subplots_adjust(wspace=0.001, hspace=0.001)

    for bidx, attn_map, imgpath in zip(range(bs), bs_attn_map, bs_image_paths):
        image = Image.open(imgpath)
        purename = os.path.basename(imgpath).split('.')[0]
        feat_size = int(math.sqrt(len(attn_map)))
        attn_map_2d = attn_map.reshape(feat_size, feat_size).detach().cpu().numpy()
        scale_factor = grid_size // attn_map_2d.shape[-1]
        ax = axes[bidx // 5, bidx % 5]  # 计算对应的行和列

        image_resized = image.resize((grid_size, grid_size))
        ax.imshow(image_resized)  # 显示原图

        expanded_map = zoom(attn_map_2d, (scale_factor, scale_factor), order=1)  # 双线性插值
        ax.imshow(expanded_map, cmap='hot', alpha=0.6, vmin=min_v, vmax=max_v)  # 使用热力图显示
        ax.axis('off')  # 关闭坐标轴
        
    plt.tight_layout()
    plt.savefig('statistic_results/WSI_heatmap/00519.png')
    plt.close()

def visual_heatmap():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    parent_infos = {}
    for imageitem in tqdm(img_annos['images'], ncols=80):
        # 'Neg/05987_0.png' -> 05987
        parent_filename = imageitem['file_name'].split('/')[-1].split('_')[0]
        annIds = coco.getAnnIds(imgIds=imageitem['id'])
        imageitem['annos'] = coco.loadAnns(annIds)
        if parent_filename not in parent_infos.keys():
            parent_infos[parent_filename] = [imageitem]
        else:
            parent_infos[parent_filename].append(imageitem)

    for p_filename, p_children in parent_infos.items():
        bs_attn_map = []
        bs_image_paths = []
        bs_gt_labels = []
        bs_pred_labels = []
        if p_filename != '00519':
            continue
        for imageitem in p_children:
            imgpath = f'{root_dir}/images/{imageitem["file_name"]}'
            # 'Neg/05987_0.png' -> 05987_0
            purename = imageitem["file_name"].split('/')[-1].split('.')[0]
            image = Image.open(imgpath)
            input_tensor = transform(image)
            databatch = {'images': input_tensor.unsqueeze(0)}
            with torch.no_grad():
                outputs = model(databatch, 'val')
            bs_attn_map.append(outputs['attn_array'][0, -1, 0, :])
            bs_image_paths.append(imgpath)
            unique_ids = list(set([ann['category_id'] for ann in imageitem['annos']]))
            if len(unique_ids) == 0:
                unique_ids = [0]
            bs_gt_labels.append(unique_ids)
            
            for confidence_pred in (outputs['pos_probs'] > 0.5).int():
                pred_labels = torch.nonzero(confidence_pred == 1)
                if len(pred_labels) == 0:
                    bs_pred_labels.append([0])
                else:
                    pred_labels = pred_labels + 1
                    bs_pred_labels.append(pred_labels.squeeze(-1).tolist())

            item_inside = [{
                'clsname': img_annos['categories'][ann['category_id']-1]['name'],
                'box_x1y1x2y2': [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]],
            } for ann in imageitem['annos']]
            img_pn = int(outputs['img_probs'].item() > 0.5)
            pos_pred = (outputs['pos_probs'][0] > 0.5).int().detach().tolist()
            if sum(pos_pred) > 0:
                img_pn = 1
            pred_result = [img_pn, *pos_pred]
            attn_array = outputs['attn_array'][0]   # (3, num_classes, num_tokens)
            num_classes, num_tokens = attn_array[0].shape
            feat_size = int(math.sqrt(num_tokens))
            attnmap_save_dir = f'{args.save_dir}/sigmoid/{p_filename}'
            os.makedirs(attnmap_save_dir, exist_ok=True)

            for idx,attn_map in enumerate(attn_array):
                attn_map = torch.sigmoid(attn_map)
                attn_map_2d = attn_map.reshape(num_classes, feat_size, feat_size)
                scale_factor = image.size[0] // attn_map_2d.shape[-1]
                save_path = f'{attnmap_save_dir}/{purename}_{idx}.png'
                draw_heatmap(attn_map_2d.detach().cpu().numpy(), image, pred_result, item_inside, scale_factor, save_path)

        bs_attn_map = torch.stack(bs_attn_map)  # (bs, num_tokens)
        os.makedirs(f'{args.save_dir}/whole_img_attn', exist_ok=True)
        save_path = f'{args.save_dir}/whole_img_attn/{p_filename}.png'
        draw_heatmap_bs(bs_attn_map, bs_image_paths, bs_gt_labels, bs_pred_labels, save_path, 0)
        # draw_heatmap_clean(bs_attn_map, bs_image_paths)
        

if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device(f'cuda:0')
    cfg = Config.fromfile(args.config_file)
    
    model = CerMCNet(
        num_classes = cfg['num_classes'], 
        backbone_type = cfg.backbone_type,
        use_lora=cfg.use_lora,
        img_size = cfg.img_size
    ).to(device)
    model.load_ckpt(args.ckpt)
    
    classnames = ['P/N', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
    root_dir = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
    json_path = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset/annofiles/OD_instances_val.json'
    coco = COCO(json_path)
    with open(json_path, 'r') as f:
        img_annos = json.load(f)
    visual_heatmap()

'''
python scripts/analyze/heatmap_v4.py \
    log/cdetector_ours/2025_02_20_23_12_21/config.py \
    log/cdetector_ours/2025_02_20_23_12_21/checkpoints/best.pth \
    statistic_results/WSI_heatmap/cdetector
'''