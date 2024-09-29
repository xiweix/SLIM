import sys
sys.path.append("../")

import os
from os.path import join as oj
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.custom_data import CustomDataset, get_transform

def get_embedding_resnet(model, loader_te, device):
    activations = dict()
    def forward_hook(module, input_batch, output):
        activations['value'] = output
        return None
    model.avgpool.register_forward_hook(forward_hook)
    embedding = []
    output_total = []
    pred_total = []
    label_total = []
    with torch.no_grad():
        for data, targets in loader_te:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            feature_vector = activations['value'].squeeze().detach().cpu()
            embedding.append(feature_vector)
            outputs = nn.Softmax(dim=1)(outputs)
            output_total.append(outputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            pred = pred.flatten().tolist()
            label = targets.flatten().tolist()
            pred_total += pred
            label_total += label
    output_np = torch.cat(output_total, dim=0).cpu().detach().numpy()
    embedding = torch.cat(embedding, dim=0).numpy()
    return embedding, output_np, pred_total, label_total

### data info
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = 'waterbirds'
base_dir = "[waterbird-data-dir]"
model_path = '[trained-model-path]'
out_dir = "../outputs/waterbirds"

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
model = getattr(models, 'resnet50')(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
test_transform = get_transform(target_resolution=(224, 224), train=False, basic_data_aug=False)
test_dataset = CustomDataset(basedir=base_dir, split="train", transform=test_transform)

loader_te = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)

### load model
### get feature vectors

feature_vector_dir = oj(out_dir, 'feature_vector')
os.makedirs(feature_vector_dir, exist_ok=True)
model = model.to(device)
net = torch.load(model_path, map_location=device)
model.load_state_dict(net['model_state_dict'])
model.eval()
fv, output, pred, label = get_embedding_resnet(model, loader_te, device)
np.save(oj(feature_vector_dir, f'train.npy'), fv)
np.save(oj(feature_vector_dir, f'train_outputs.npy'), output)
np.save(oj(feature_vector_dir, f'train_predictions.npy'), pred)
np.save(oj(feature_vector_dir, f'train_labels.npy'), label)