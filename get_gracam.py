import sys
sys.path.append("../")

import os
from os.path import join as oj
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from utils.gradcam import GradCAM
from utils.evaluation import get_att_eval, get_iou
from tqdm import tqdm
from PIL import Image
import matplotlib.cm
from copy import deepcopy
import torchvision
import torch
import torch.nn as nn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = 'waterbirds'
base_dir = "[waterbird-data-dir]"
method_name = 'ERM'
model_path = '[trained-model-path].pt'
out_dir = "../outputs/waterbirds"

img_dir = oj(base_dir, 'waterbird_complete95_forest2water2')
model_name = 'resnet50'

bbox_path = oj(base_dir, 'bounding_boxes.txt')
bbox_info = pd.read_csv(bbox_path, sep=' ', names=['img_id', 'xmin', 'ymin', 'width', 'height'])

test_df = pd.read_csv(oj(base_dir, 'test.csv'))
test_df['index'] = test_df.index
img_id_list = test_df['img_id'].tolist()
bbox_test = bbox_info.query('img_id in @img_id_list').reset_index(drop=True)
test_df = pd.concat([test_df, bbox_test.drop(['img_id'], axis=1)], axis=1)

total_label = test_df['y'].values
total_place = test_df['place'].values
meta_info_str = np.array([f"label_{total_label[i]}.place_{total_place[i]}" for i in range(len(total_label))])

img_folder_list = test_df["img_folder"].tolist()
img_name_list = test_df["img_name"].tolist()

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.Resize((224, 224)),
])
model = getattr(models, 'resnet50')(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

n_classes = 2

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d
    def forward(self, x):
        return x
def initialize_torchvision_model(name, d_out, **kwargs):
    if name == 'resnet50':
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:
        last_layer = Identity(d_features)
        model.d_out = d_features
    else:
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)
    return model


name = 'resnet50'  # Example model name
d_out = 2
model = initialize_torchvision_model(name, d_out, pretrained=False)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['algorithm']
adjusted_state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
model.load_state_dict(adjusted_state_dict)
model.to(device)
model.eval()

layer_name = 'layer4'
model_dict = dict(type=model_name, arch=model, layer_name=layer_name, input_size=(224, 244), device=device)
gradcam = GradCAM(model_dict, True)
gradcam_dir = oj(out_dir, 'gradCAM')
os.makedirs(gradcam_dir, exist_ok=True)
print('\n\n', gradcam_dir, '\n\n')

aiou_seg_list = []
iou_seg_list = []
aiou_bbox_list = []
iou_bbox_list = []
for i in tqdm(range(len(img_name_list))):
    cur_img_name = img_name_list[i]
    cur_img_path = os.path.join(img_dir, img_folder_list[i], cur_img_name)
    cur_img = Image.open(cur_img_path).convert('RGB')
    w, h = cur_img.size
    cur_batch = preprocess(cur_img).unsqueeze(0).to(device)
    img_sized_mask, _, _, _, _, _ = gradcam(cur_batch, out_size=(h, w))     
    img_sized_mask = img_sized_mask.squeeze().cpu().numpy()
    img_sized_mask = np.nan_to_num(img_sized_mask)
    copied_img_sized_mask = deepcopy(img_sized_mask)
    segment_dir = img_dir.replace('waterbird_complete95_forest2water2', 'segmentation_masks')
    seg = np.load(oj(segment_dir, cur_img_name.replace('jpg', 'npy')))
    
    aiou_seg = get_att_eval(img_sized_mask, gt_mask=seg)
    iou_seg = get_iou(img_sized_mask, gt_mask=seg)
    
    xmin, ymin, width, height = test_df.iloc[[i]].values.tolist()[0][-4:]
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin+width), int(ymin+height)
    aiou_bbox = get_att_eval(img_sized_mask, xmin, ymin, xmax, ymax)
    iou_bbox = get_iou(img_sized_mask, xmin, ymin, xmax, ymax)
    
    aiou_seg_list.append(aiou_seg)
    iou_seg_list.append(iou_seg)
    aiou_bbox_list.append(aiou_bbox)
    iou_bbox_list.append(iou_bbox)
    
    heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_img_sized_mask) * 255))
    heatmapWimg = deepcopy(cur_img)
    heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_img_sized_mask.shape)*0.5*255)))
    heatmap = heatmap.convert('RGB')
    heatmapWimg = heatmapWimg.convert('RGB')
    heatmap.save(os.path.join(gradcam_dir, f'{cur_img_name}_gradcam.jpg'))
    heatmapWimg.save(os.path.join(gradcam_dir, f'{cur_img_name}_gradcam_img.jpg'))
np.save(os.path.join(out_dir, 'aiou_seg_list.npy'), aiou_seg_list)
np.save(os.path.join(out_dir, 'iou_seg_list.npy'), iou_seg_list)
np.save(os.path.join(out_dir, 'aiou_bbox_list.npy'), aiou_bbox_list)
np.save(os.path.join(out_dir, 'iou_bbox_list.npy'), iou_bbox_list)


def get_eval(att_eval_list, meta_info_str):
    att_eval_list = np.array(att_eval_list)
    ### total
    total_att = np.mean(att_eval_list)
    ### subgroup
    sub_group_att = {}
    worst_att = 100.
    for i in range(2):
        for j in range(2):
            subgroup_name = f"label_{i}.place_{j}"
            indices = np.where(meta_info_str==subgroup_name)[0]
            group_att = np.mean(att_eval_list[indices])
            sub_group_att[subgroup_name] = group_att
            if worst_att > group_att:
                worst_att = group_att
    return total_att, sub_group_att, worst_att

for name, att in zip(['aiou_seg_list', 'iou_seg_list', 'aiou_bbox_list', 'iou_bbox_list'], [aiou_seg_list, iou_seg_list, aiou_bbox_list, iou_bbox_list]):
    total_att, sub_group_att, worst_att = get_eval(att, meta_info_str)
    print("\n\n\n")
    print(name)
    print(f"total_att: {total_att}")
    print(f"sub_group_att: {sub_group_att}")
    print(f"worst_att: {worst_att}")
    with open(oj(out_dir, 'attention_evaluation.txt'), 'a') as f:
        f.write(f"\n\nname: {name}\n")
        f.write(f"total_att: {total_att}\n")
        f.write(f"sub_group_att: {sub_group_att}\n")
        f.write(f"worst_att: {worst_att}\n")
