import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from utils.gradcam import GradCAM
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.cm
from copy import deepcopy
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_results(acc_groups, get_yp_func):
    groups = acc_groups.keys()
    results = {
            f"accuracy_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
            for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    results.update({"worst_accuracy" : min(results.values())})
    return results

def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)

def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p

def evaluate(model, loader, get_yp_func):
    model.eval()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(loader.dataset.n_groups)}
    with torch.no_grad():
        for _, x, y, g, p in tqdm(loader):
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            logits = model(x)
            update_dict(acc_groups, y, g, logits)
    model.train()
    return get_results(acc_groups, get_yp_func)

def get_iou(attention, xmin=None, ymin=None, xmax=None, ymax=None, gt_mask=None):
    if gt_mask is None:
        gt_mask = np.zeros((attention.shape))
        gt_mask[xmin:xmax, ymin:ymax] = 1
    intersection = (np.minimum(gt_mask, attention)).sum()
    union = (np.maximum(gt_mask, attention)).sum()
    iou_score = intersection / union
    return iou_score

def get_att_eval(attention, xmin=None, ymin=None, xmax=None, ymax=None, gt_mask=None):
    if gt_mask is None:
        gt_mask = np.zeros((attention.shape))
        gt_mask[xmin:xmax, ymin:ymax] = 1
    delta = 0.8 ## 0.9
    m1 = gt_mask * attention
    m2 = attention
    m1[m1 < delta] = 0.
    m2[m2 < delta] = 0.
    try:
        score = (m1.sum()) / (m2.sum())
    except RuntimeWarning:
        score = 0.
    if math.isnan(score):
        score = 0.
    return score

def compute_accuracy_and_loss(model, data_loader, device):
    model.eval()
    num_examples = 0
    output_total = []
    pred_total = []
    label_total = []
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            targets = targets.long()
            num_examples += targets.size(0)
            outputs = nn.Softmax(dim=1)(outputs)
            output_total.append(outputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            pred = pred.flatten().tolist()
            label = targets.flatten().tolist()
            pred_total += pred
            label_total += label
    output_np = torch.cat(output_total, dim=0).cpu().detach().numpy()
    val_acc = accuracy_score(pred_total, label_total)
    val_f1 = f1_score(pred_total, label_total)
    return val_acc, val_f1, output_np, pred_total, label_total

def compute_performance(model_name, model, layer_name, data_loader, device, img_dir, out_dir, gt=False):
    model.eval()
    model_dict = dict(type=model_name, arch=model, layer_name=layer_name, input_size=(224, 224), device=device)
    gradcam = GradCAM(model_dict, True)
    output_total = []
    pred_total = []
    label_total = []
    place_total = []
    att_seg_total = []
    confidence_total = []
    total_vectors = []
    total_masks = []
    total_entropy_score = []
    for i, (_, img, label, g, place, img_name, img_path) in enumerate(tqdm(data_loader)):
        img = img.to(device)
        img_name = img_name[0]
        label = int(label[0])
        img_path = img_path[0]
        pil_img = Image.open(img_path).convert('RGB')
        w, h = pil_img.size
        if gt is True:
            img_sized_mask, outputs, _, small_mask, _, activations = gradcam(img, class_idx=label, out_size=(h, w))
        else:       
            img_sized_mask, outputs, _, small_mask, _, activations = gradcam(img, class_idx=None, out_size=(h, w))                 
        img_sized_mask = np.nan_to_num(img_sized_mask.squeeze().cpu().numpy())
        copied_img_sized_mask = deepcopy(img_sized_mask)
        os.makedirs(out_dir, exist_ok=True)
        heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_img_sized_mask) * 255.))
        heatmapWimg = deepcopy(pil_img)
        heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_img_sized_mask.shape)*0.5*255.)))
        heatmap = heatmap.convert('RGB')
        heatmapWimg = heatmapWimg.convert('RGB')
        heatmap.save(os.path.join(out_dir, f'{img_name}_gradcam.jpg'))
        heatmapWimg.save(os.path.join(out_dir, f'{img_name}_gradcam_img.jpg'))
        
        small_mask = small_mask.detach().squeeze().cpu().numpy()
        small_mask = np.nan_to_num(small_mask)
        vector = activations.detach().squeeze().cpu().numpy()
        vector_mask_temp = np.expand_dims(small_mask, axis=0)
        vector_mask_expand = np.repeat(vector_mask_temp, vector.shape[0], axis=0)
        res = np.sum(vector * vector_mask_expand, axis=(1, 2)) ### weighted sum
        total_vectors.append(res)
        total_masks.append(small_mask)
        np.save(os.path.join(out_dir, f'{img_name}_small_mask.npy'), small_mask)
        np.save(os.path.join(out_dir, f'{img_name}_weighted_fv.npy'), res)
        
        entropy_score = torch.nn.functional.softmax(outputs, dim=1)
        entropy_score = entropy_score * torch.log2(entropy_score)
        entropy_score = (-1*torch.sum(entropy_score, dim=1)).detach().cpu().numpy()[0]  ### larger -- higher entropy
        total_entropy_score.append(entropy_score)
        
        segment_dir = img_dir.replace('waterbird_complete95_forest2water2', 'segmentation_masks')
        seg = np.load(os.path.join(segment_dir, img_name.replace('jpg', 'npy')))
        att = get_att_eval(img_sized_mask, gt_mask=seg)
        att_seg_total.append(att)
        outputs = nn.Softmax(dim=1)(outputs)
        output_total.append(outputs)
        pred = outputs.argmax(dim=1, keepdim=True)
        pred = pred.flatten().tolist()
        pred_total += pred
        label_total.append(label)
        place_total += place.flatten().tolist()
    output_np = torch.cat(output_total, dim=0).cpu().detach().numpy()
    total_vectors = np.array(total_vectors)
    total_masks_flatten = np.array(total_masks)
    total_masks_flatten = total_masks_flatten.reshape((total_masks_flatten.shape[0], -1))    

    for i in range(len(pred_total)):
        c = output_np[i, pred_total[i]]
        confidence_total.append(c)
    acc = accuracy_score(pred_total, label_total)
    f1 = f1_score(pred_total, label_total)
    return acc, f1, output_np, confidence_total, pred_total, label_total, place_total, total_entropy_score, total_vectors, total_masks_flatten, att_seg_total


def compute_accuracy_and_attention(model_name, model, layer_name, data_loader, device, img_dir, df, episode, out_dir, num_classes=2):
    all_classes = list(range(num_classes))
    model.eval()
    model_dict = dict(type=model_name, arch=model, layer_name=layer_name, input_size=(224, 224), device=device)
    gradcam = GradCAM(model_dict, True)
    output_total = []
    pred_total = []
    label_total = []
    place_total = []
    iou_seg_total = []
    aiou_seg_total = []
    iou_bbox_total = []
    aiou_bbox_total = []
    att_seg_total = []
    att_bbox_total = []
    confidence_total = []
    total_vectors = []
    total_masks = []
    total_entropy_score = []
    for i, (img, label, place, img_name, img_path) in enumerate(tqdm(data_loader)):
        img = img.to(device)
        img_name = img_name[0]
        label = int(label[0])
        img_path = img_path[0]
        pil_img = Image.open(img_path).convert('RGB')
        w, h = pil_img.size
        # img_sized_mask, outputs, _, small_mask, _, activations = gradcam(img, class_idx=label, out_size=(h, w))       
        img_sized_mask, outputs, _, small_mask, _, activations = gradcam(img, class_idx=None, out_size=(h, w))                 
        img_sized_mask = np.nan_to_num(img_sized_mask.squeeze().cpu().numpy())
        copied_img_sized_mask = deepcopy(img_sized_mask)
        os.makedirs(out_dir, exist_ok=True)
        heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_img_sized_mask) * 255.))
        heatmapWimg = deepcopy(pil_img)
        heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_img_sized_mask.shape)*0.5*255.)))
        heatmap = heatmap.convert('RGB')
        heatmapWimg = heatmapWimg.convert('RGB')
        heatmap.save(os.path.join(out_dir, f'{img_name}_gradcam.jpg'))
        heatmapWimg.save(os.path.join(out_dir, f'{img_name}_gradcam_img.jpg'))
        
        small_mask = small_mask.detach().squeeze().cpu().numpy()
        small_mask = np.nan_to_num(small_mask)
        vector = activations.detach().squeeze().cpu().numpy()
        vector_mask_temp = np.expand_dims(small_mask, axis=0)
        vector_mask_expand = np.repeat(vector_mask_temp, vector.shape[0], axis=0)
        res = np.sum(vector * vector_mask_expand, axis=(1, 2)) ### weighted sum
        total_vectors.append(res)
        total_masks.append(small_mask)
        np.save(os.path.join(out_dir, f'{img_name}_small_mask.npy'), small_mask)
        np.save(os.path.join(out_dir, f'{img_name}_weighted_fv.npy'), res)
        
        entropy_score = torch.nn.functional.softmax(outputs, dim=1)
        entropy_score = entropy_score * torch.log2(entropy_score)
        entropy_score = (-1*torch.sum(entropy_score, dim=1)).detach().cpu().numpy()[0]  ### larger -- higher entropy
        total_entropy_score.append(entropy_score)
        
        segment_dir = img_dir.replace('waterbird_complete95_forest2water2', 'segmentation_masks')
        seg = np.load(os.path.join(segment_dir, img_name.replace('jpg', 'npy')))
        iou = get_iou(img_sized_mask, gt_mask=seg)
        xmin, ymin, width, height = df.iloc[[i]].values.tolist()[0][-4:]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin+width), int(ymin+height)
        iou_bbox = get_iou(img_sized_mask, xmin, ymin, xmax, ymax)
        att = get_att_eval(img_sized_mask, gt_mask=seg)
        att_bbox = get_att_eval(img_sized_mask, xmin, ymin, xmax, ymax)
        max_other_iou = 0
        max_other_iou_bbox = 0
        for l in all_classes:
            if l != label:
                temp_att_mask, _, _, _, _, _ = gradcam(img, class_idx=l, out_size=(h, w))            
                temp_att_mask = np.nan_to_num(temp_att_mask.squeeze().cpu().numpy())
                temp_iou = get_iou(temp_att_mask, gt_mask=seg)
                temp_iou_bbox = get_iou(temp_att_mask, xmin, ymin, xmax, ymax)
                max_other_iou = temp_iou if temp_iou >= max_other_iou else max_other_iou
                max_other_iou_bbox = temp_iou_bbox if temp_iou_bbox >= max_other_iou_bbox else max_other_iou_bbox
        aiou = iou / (iou+max_other_iou)
        aiou_bbox = iou_bbox / (iou_bbox+max_other_iou_bbox)
        aiou_seg_total.append(aiou)
        aiou_bbox_total.append(aiou_bbox)
        iou_seg_total.append(iou)
        iou_bbox_total.append(iou_bbox)
        att_seg_total.append(att)
        att_bbox_total.append(att_bbox)
        outputs = nn.Softmax(dim=1)(outputs)
        output_total.append(outputs)
        pred = outputs.argmax(dim=1, keepdim=True)
        pred = pred.flatten().tolist()
        pred_total += pred
        label_total.append(label)
        place_total += place.flatten().tolist()

    output_np = torch.cat(output_total, dim=0).cpu().detach().numpy()
    total_vectors = np.array(total_vectors)
    total_masks_flatten = np.array(total_masks)
    total_masks_flatten = total_masks_flatten.reshape((total_masks_flatten.shape[0], -1))    

    for i in range(len(pred_total)):
        c = output_np[i, pred_total[i]]
        confidence_total.append(c)
    acc = accuracy_score(pred_total, label_total)
    f1 = f1_score(pred_total, label_total)
    return acc, f1, output_np, confidence_total, pred_total, label_total, place_total, total_entropy_score, total_vectors, total_masks_flatten, iou_seg_total, iou_bbox_total, aiou_seg_total, aiou_bbox_total, att_seg_total, att_bbox_total