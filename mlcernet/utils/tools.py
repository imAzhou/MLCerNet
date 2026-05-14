import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    one_million = 1e6
    return {
        'Total': f'{(total_num / one_million):.4f}M',
        'Trainable': f'{(trainable_num / one_million):.4f}M',
    }


def is_bbox_inside(bbox1, bbox2, tolerance=0):
    return (
        bbox1[0] >= bbox2[0] - tolerance
        and bbox1[1] >= bbox2[1] - tolerance
        and bbox1[2] <= bbox2[2] + tolerance
        and bbox1[3] <= bbox2[3] + tolerance
    )


def generate_cut_regions(region_start, region_width, region_height, k, stride=400, minlen=0):
    x_start, y_start = region_start
    overlap = k - stride
    cut_regions = []

    exact_w = (region_width // stride) * stride + overlap
    exact_h = (region_height // stride) * stride + overlap

    w_rem = region_width - exact_w
    h_rem = region_height - exact_h
    end_x = exact_w if w_rem > minlen else exact_w - stride
    end_y = exact_h if h_rem > minlen else exact_h - stride

    for x in range(0, end_x, stride):
        for y in range(0, end_y, stride):
            x1, y1 = x, y
            x2, y2 = x1 + k, y1 + k

            if x2 > region_width:
                x2 = region_width
                x1 = x2 - k
            if y2 > region_height:
                y2 = region_height
                y1 = y2 - k

            cut_regions.append([x1 + x_start, y1 + y_start, x2 + x_start, y2 + y_start])
    return cut_regions


def draw_OD(read_image, save_path, square_coords, inside_items, class_labels):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_labels)))[:, :3] * 255
    category_colors = {cat: tuple(map(int, color)) for cat, color in zip(class_labels, colors)}

    draw = ImageDraw.Draw(read_image)
    sq_x1, sq_y1, sq_w, sq_h = square_coords

    for box_item in inside_items:
        category = box_item.get('sub_class')
        x1, y1, x2, y2 = box_item.get('region')
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1 + sq_w, x2) - sq_x1
        y_max = min(sq_y1 + sq_h, y2) - sq_y1
        color = category_colors.get(category, (255, 255, 255))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 2, y_min - 15), category, fill=color)

    fig, ax = plt.subplots(figsize=(sq_w // 100 + 1, sq_h // 100 + 1), dpi=100)
    ax.imshow(np.array(read_image))
    ax.axis('off')
    patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=category)
        for category, color in category_colors.items()
    ]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.02, 1), frameon=False)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
