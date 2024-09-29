import sys
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_auc_score, f1_score

def get_time():
    # Get the current time with microsecond precision
    now = datetime.now()
    
    # Format the string to include year, month, day, hour, minute, second, and microsecond
    time_str = f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}_{now.microsecond:04d}"
    
    return time_str

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


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


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p

def update_dict(acc_groups, auroc_f1_groups, y, g, p, logits, preds_prob, preds_prob_sigmoid):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g, y, p = g.cpu(), y.cpu(), p.cpu()
    preds, preds_prob, preds_prob_sigmoid = preds.cpu(), preds_prob.cpu(), preds_prob_sigmoid.cpu()
    
    # Update for entire dataset
    auroc_f1_groups['all']['labels'].extend(y.tolist())
    auroc_f1_groups['all']['preds_prob'].extend(preds_prob.tolist())
    auroc_f1_groups['all']['preds_prob_sigmoid'].extend(preds_prob_sigmoid.tolist())
    auroc_f1_groups['all']['preds'].extend(preds.tolist())

    # Update for no_patch subgroup
    no_patch_mask = p == 0
    auroc_f1_groups['no_patch']['labels'].extend(y[no_patch_mask].tolist())
    auroc_f1_groups['no_patch']['preds_prob'].extend(preds_prob[no_patch_mask].tolist())
    auroc_f1_groups['no_patch']['preds_prob_sigmoid'].extend(preds_prob_sigmoid[no_patch_mask].tolist())
    auroc_f1_groups['no_patch']['preds'].extend(preds[no_patch_mask].tolist())

    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)

def write_dict_to_tb(writer, dict, prefix, step):
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)

def get_results(acc_groups, all_f1, all_auroc, all_auroc_sigmoid, all_accuracy, no_patch_f1, no_patch_auroc, no_patch_auroc_sigmoid, no_patch_accuracy, get_yp_func, loss):
    groups = acc_groups.keys()
    results = {
        f"accuracy_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
        for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"worst_accuracy" : min(results.values())})
    results.update({"mean_accuracy": np.mean(list(results.values()))})
    results.update({"all_accuracy": all_accuracy})
    results.update({"all_accuracy_2": all_correct/all_total})
    results.update({"all_f1": all_f1})
    results.update({"all_auroc": all_auroc})
    results.update({"all_auroc_sigmoid": all_auroc_sigmoid})
    results.update({"no_patch_accuracy": no_patch_accuracy})
    results.update({"no_patch_f1": no_patch_f1})
    results.update({"no_patch_auroc": no_patch_auroc})
    results.update({"no_patch_auroc_sigmoid": no_patch_auroc_sigmoid})
    results.update({"loss": loss})
    return results

def evaluate(model, loader, get_yp_func, args):
    criterion = torch.nn.CrossEntropyLoss().to(f"cuda:{args.gpu}")
    model.eval()
    loss_meter = AverageMeter()
    acc_groups = {g_idx: AverageMeter() for g_idx in range(loader.dataset.n_groups)}
    auroc_f1_groups = {
        'all': {'labels': [], 'preds_prob': [], 'preds_prob_sigmoid': [], 'preds': []},
        'no_patch': {'labels': [], 'preds_prob': [], 'preds_prob_sigmoid': [], 'preds': []}
    }
    with torch.no_grad():
        for x, y, g, p in tqdm.tqdm(loader):
            x, y, p = x.to(f"cuda:{args.gpu}"), y.to(f"cuda:{args.gpu}"), p.to(f"cuda:{args.gpu}")
            logits = model(x)
            loss = criterion(logits, y)
            loss_meter.update(loss, x.size(0))
            preds_prob = torch.softmax(logits, dim=1)[:, 1]  # binary classification
            preds_prob_sigmoid = torch.sigmoid(logits)[:, 1]
            update_dict(acc_groups, auroc_f1_groups, y, g, p, logits, preds_prob, preds_prob_sigmoid)
    
    f_loss = loss_meter.avg.detach().cpu().numpy()
    all_labels = np.array(auroc_f1_groups['all']['labels'])
    all_preds_prob = np.array(auroc_f1_groups['all']['preds_prob'])
    all_preds_prob_sigmoid = np.array(auroc_f1_groups['all']['preds_prob_sigmoid'])
    all_preds = np.array(auroc_f1_groups['all']['preds'])
    all_f1 = f1_score(all_labels, all_preds)
    all_auroc = roc_auc_score(all_labels, all_preds_prob)
    all_auroc_sigmoid = roc_auc_score(all_labels, all_preds_prob_sigmoid)
    all_accuracy = (all_preds == all_labels).mean() 
    no_patch_labels = np.array(auroc_f1_groups['no_patch']['labels'])
    no_patch_preds_prob = np.array(auroc_f1_groups['no_patch']['preds_prob'])
    no_patch_preds_prob_sigmoid = np.array(auroc_f1_groups['no_patch']['preds_prob_sigmoid'])
    no_patch_preds = np.array(auroc_f1_groups['no_patch']['preds'])
    no_patch_f1 = f1_score(no_patch_labels, no_patch_preds)
    no_patch_auroc = roc_auc_score(no_patch_labels, no_patch_preds_prob)
    no_patch_auroc_sigmoid = roc_auc_score(no_patch_labels, no_patch_preds_prob_sigmoid)
    no_patch_accuracy = (no_patch_preds == no_patch_labels).mean()
    
    model.train()
    return get_results(acc_groups, all_f1, all_auroc, all_auroc_sigmoid, all_accuracy, no_patch_f1, no_patch_auroc, no_patch_auroc_sigmoid, no_patch_accuracy, get_yp_func, f_loss)


class MultiTaskHead(nn.Module):
    def __init__(self, n_features, n_classes_list):
        super(MultiTaskHead, self).__init__()
        self.fc_list = [nn.Linear(n_features, n_classes) for n_classes in n_classes_list]

    def forward(self, x):
        outputs = []
        for head in self.fc_list:
            out = head(x)
            outputs.append(out)
        return outputs

def allocate_sampling_sizes(density_list, total_budget):
    # Calculate inverse weights (higher density => smaller weight)
    weights = [1/density for density in density_list if density > 0]

    # Normalize weights to sum up to 1
    total_weight = sum(weights)
    normalized_weights = [weight/total_weight for weight in weights]

    # Allocate total budget according to normalized weights
    sampling_sizes = [int(round(total_budget * weight)) for weight in normalized_weights]

    return sampling_sizes

def compute_cluster_density(embeddings, labels):
    # centroids = kmeans_model.cluster_centers_
    # labels = kmeans_model.labels_
    unique_labels = np.unique(labels)
    densities = []
    for label in unique_labels:
        # Extract data for the current cluster
        data_in_cluster = embeddings[labels == label]
        centroid = np.mean(data_in_cluster, axis=0)
        # Calculate the average Euclidean distance from each point to the centroid
        distances = np.sqrt(np.sum((data_in_cluster - centroid) ** 2, axis=1))
        # Compute the average distance (inverse to represent density)
        average_distance = np.mean(distances)
        density = 1 / average_distance if average_distance != 0 else 0
        densities.append(density)
    return np.array(densities)

def compute_sampling_probability(embeddings, labels):
    unique_labels = np.unique(labels)
    density_factors = []
    for label in unique_labels:
        data_in_cluster = embeddings[labels == label]
        centroid = np.mean(data_in_cluster, axis=0)
        all_distances = np.sqrt(np.sum((data_in_cluster - centroid) ** 2, axis=1))
        median_distance = np.median(all_distances)
        density_factors.append(median_distance)
    density_factors = np.array(density_factors) / np.sum(density_factors)
    print("\n")
    print("density_factors: ", density_factors)
    print("\n")
    sampling_probability = density_factors
    return sampling_probability