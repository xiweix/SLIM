"""Evaluate DFR on spurious correlations datasets."""
import sys
sys.path.append("../")

import torch
import torchvision
import numpy as np
import os
import tqdm
import argparse
import sys
from functools import partial
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.wb_data import WaterBirdsDataset, get_loader, get_transform_cub
from utils.utils import Logger, set_seed
from utils.analysis import get_time
from sklearn.metrics import roc_auc_score, f1_score
import json

# WaterBirds
C_OPTIONS = [1., 0.7, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.003]
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 5., 7., 10., 20., 50., 80., 100., 200., 300., 500., 1000.]

REG = "l1"
CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [{0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]

parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="", help="Checkpoint path")
parser.add_argument(
    "--exp_info",
    type=str,
    default="",
    help="Checkpoint info")
parser.add_argument(
    "--val_split_path",
    type=str,
    default=None)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    required=False,
    help="batch size")
parser.add_argument(
    "--multi_run",
    type=int,
    default=0,
    required=False,
    help="Run multiple times in the same setting")
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    required=False,
    help="Set Random Seed | default = 0")
parser.add_argument(
    "--dataset",
    type=str,
    default="waterbirds",
    help="dataset name")
args = parser.parse_args()

def dfr_tune(valIndex, out_dir, flag='dfr', balance_val=False, preprocess=True, num_retrains=1, ratio=0.5):    
    worst_accs = {}
    for i in range(num_retrains):
        set_seed(args.seed + i)
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_val = int(len(valIndex) * ratio)
        np.random.shuffle(valIndex)
        valTuneIndex = valIndex[n_val:]
        x_valtrain = x_val[valTuneIndex]
        y_valtrain = y_val[valTuneIndex]
        g_valtrain = g_val[valTuneIndex]
        n_groups = len(np.unique(g_valtrain))
        g_idx = [np.where(g_valtrain == g)[0] for g in np.unique(g_valtrain)]
        min_g = np.min([len(g) for g in g_idx])
        if i == 0:
            logger.write(f"valTuneIndex: {len(valTuneIndex)}\n")
            logger.write(f"valtrain size ({len(x_valtrain)})\n")
            logger.write(f"n_groups: {n_groups}\n")
            logger.write(f"min_g: {min_g}\n")
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:  ### requires meta info
            temp_index = np.concatenate([g[:min_g] for g in g_idx])
            x_valtrain = x_valtrain[temp_index]
            y_valtrain = y_valtrain[temp_index]
            g_valtrain = g_valtrain[temp_index]
            if i == 0:
                logger.write(f"Balance meta info: valtrain size ({len(x_valtrain)})\n")
        n_train = len(x_valtrain)
        if i == 0:
            logger.write(f"n_train: {n_train}\n")
        if ratio == 0.:
            x_valval = x_valtrain
            y_valval = y_valtrain
            g_valval = g_valtrain
        else:
            x_valval = x_val[valIndex[:n_val]]
            y_valval = y_val[valIndex[:n_val]]
            g_valval = g_val[valIndex[:n_val]]
        if i == 0:
            logger.write(f"valval size ({len(x_valval)})\n")
            logger.write(f"g_valtrain: {np.bincount(g_valtrain)}\n")
            logger.write(f"x_valtrain: {len(x_valtrain)}\n")
        if preprocess:
            scaler = StandardScaler()
            x_valtrain = scaler.fit_transform(x_valtrain)
            x_valval = scaler.transform(x_valval)
        if balance_val:
            cls_w_options = [{0: 1., 1: 1.}]
            if i == 0:
                logger.write(f"cls_w_options: {cls_w_options}\n")
        else:
            cls_w_options = CLASS_WEIGHT_OPTIONS
            if i == 0:
                logger.write(f"cls_w_options: {cls_w_options}\n")
        for c in C_OPTIONS:
            for class_weight in cls_w_options:
                logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                            class_weight=class_weight)
                logreg.fit(x_valtrain, y_valtrain)
                preds_val = logreg.predict(x_valval)
                group_accs = np.array([(preds_val == y_valval)[g_valval == g].mean() for g in np.unique(g_valtrain)])
                worst_acc = np.min(group_accs)
                if i == 0:
                    worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                else:
                    worst_accs[c, class_weight[0], class_weight[1]] += worst_acc
                # print(f"{c}, {class_weight}, {worst_acc}, {group_accs}, {worst_accs[c, class_weight[0], class_weight[1]]}")
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers

def dfr_eval(c, w1, w2, all_embeddings, all_y, all_g, all_p, logger, val_index, out_dir, flag='dfr', num_retrains=20, preprocess=True, balance_val=False):
    logger.write(f"====================={flag}===dfr_eval====================\n")
    logger.write("params:\n")
    logger.write(f"c: {c}\n")
    logger.write(f"w1: {w1}\n")
    logger.write(f"w2: {w2}\n")
    logger.write(f"flag: {flag}\n")
    logger.write(f"preprocess: {preprocess}\n")
    logger.write(f"balance_val: {balance_val}\n")
    logger.write(f"num_retrains: {num_retrains}\n")

    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["val"])

    for i in range(num_retrains):
        set_seed(args.seed + i)
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        p_val = all_p["val"]
        # logger.write(f"DFR: Select data size from validation set: {len(val_index)}\n")
        x_val = x_val[val_index]
        y_val = y_val[val_index]
        g_val = g_val[val_index]
        p_val = p_val[val_index]
        n_groups = len(np.unique(g_val))
        # logger.write(f"n_groups: {n_groups}\n")
        g_idx = [np.where(g_val == g)[0] for g in np.unique(g_val)]
        min_g = np.min([len(g) for g in g_idx])
        # logger.write(f"min_g: {min_g}\n")
        for g in g_idx:
            np.random.shuffle(g)
        org_data_size = len(val_index)
        if balance_val:  ### requires meta info
            temp_index = np.concatenate([g[:min_g] for g in g_idx])
            x_val = x_val[temp_index]
            y_val = y_val[temp_index]
            g_val = g_val[temp_index]
            p_val = p_val[temp_index]
            # logger.write(f"Balance meta info: val size ({len(x_val)})\n")
        x_train = x_val
        y_train = y_val
        g_train = g_val
        p_train = p_val
        # logger.write(f"g_train: {len(g_train)}, {np.bincount(g_train)}\n")
        # logger.write(f"p_train: {len(p_train)}, {np.bincount(p_train)}\n")
        # logger.write(f"x_train: {len(x_train)}\n")
        if preprocess:
            x_train = scaler.transform(x_train)
        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                    class_weight={0: w1, 1: w2})
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    p_test = all_p["test"]
    logger.write(f"g_test: {np.bincount(g_test)}\n")
    logger.write(f"p_test: {np.bincount(p_test)}\n")
    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                class_weight={0: w1, 1: w2})
    n_classes = len(np.unique(y_train))
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    current_results = {}
    
    preds_test = logreg.predict(x_test)
    probs_test = logreg.predict_proba(x_test)[:, 1]
    test_accs = [(preds_test == y_test)[g_test == g].mean() for g in np.unique(g_val)]
    test_mean_acc = (preds_test == y_test).mean()
    test_auroc = roc_auc_score(y_test, probs_test)
    test_f1 = f1_score(y_test, preds_test)
    ### no_patch
    test_no_patch_index = p_test == 0
    test_no_patch_acc = (preds_test == y_test)[test_no_patch_index].mean()
    test_no_patch_auroc = roc_auc_score(y_test[test_no_patch_index], probs_test[test_no_patch_index])
    test_no_patch_f1 = f1_score(y_test[test_no_patch_index], preds_test[test_no_patch_index])
    
    preds_train = logreg.predict(x_train)
    probs_train = logreg.predict_proba(x_train)[:, 1]
    train_accs = [(preds_train == y_train)[g_train == g].mean() for g in np.unique(g_val)]
    train_mean_acc = (preds_train == y_train).mean()
    train_auroc = roc_auc_score(y_train, probs_train)
    train_f1 = f1_score(y_train, preds_train)
    ### no_patch
    train_no_patch_index = p_train == 0
    train_no_patch_acc = (preds_train == y_train)[train_no_patch_index].mean()
    train_no_patch_auroc = roc_auc_score(y_train[train_no_patch_index], probs_train[train_no_patch_index])
    train_no_patch_f1 = f1_score(y_train[train_no_patch_index], preds_train[train_no_patch_index])
    org_data_size, len(x_train)
    
    current_results["data_size"] = org_data_size
    current_results["train_data_size"] = len(x_train)
    
    current_results["train_accs"] = train_accs
    current_results["train_worst_acc"] = np.min(train_accs)
    current_results["train_mean_acc"] = train_mean_acc
    current_results["train_auroc"] = train_auroc
    current_results["train_f1"] = train_f1
    current_results["train_no_patch_acc"] = train_no_patch_acc
    current_results["train_no_patch_auroc"] = train_no_patch_auroc
    current_results["train_no_patch_f1"] = train_no_patch_f1
    
    current_results["test_accs"] = test_accs
    current_results["test_worst_acc"] = np.min(test_accs)
    current_results["test_mean_acc"] = test_mean_acc
    current_results["test_auroc"] = test_auroc
    current_results["test_f1"] = test_f1
    current_results["test_no_patch_acc"] = test_no_patch_acc
    current_results["test_no_patch_auroc"] = test_no_patch_auroc
    current_results["test_no_patch_f1"] = test_no_patch_f1
    return current_results

set_seed(args.seed)
cur_time = get_time()

valIndex = np.load(args.val_split_path)
file_name = args.val_split_path.split('/')[-1]
result_dir = "results"
out_dir = os.path.join(result_dir, f"{args.exp_info}_{cur_time}_{file_name}")
out_dir += f"_run_{args.multi_run}"
os.makedirs(out_dir, exist_ok=True)
print(f"save results in: ", out_dir)
logger = Logger(os.path.join(out_dir, 'log.txt'))

with open(os.path.join(out_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
with open(os.path.join(out_dir, 'args.json'), 'w') as f:
    args_json = json.dumps(vars(args))
    f.write(args_json)

## Load data
target_resolution = (224, 224)
train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=False)
test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)

if args.dataset == 'celebA':
    logger.write(f"load {args.dataset} data\n")
    data_dir = 'celebA'
    img_folder='img_align_celeba'
elif args.dataset == 'waterbirds':
    logger.write(f"load {args.dataset} data\n")
    data_dir = 'waterbird'
    img_folder='waterbird_complete95_forest2water2'
elif args.dataset == 'isic':
    logger.write(f"load {args.dataset} data\n")
    data_dir = 'ISIC'
    img_folder='img'
trainset = WaterBirdsDataset(basedir=data_dir, split="train", transform=train_transform, img_folder=img_folder)
testset = WaterBirdsDataset(basedir=data_dir, split="test", transform=test_transform, img_folder=img_folder)
valset = WaterBirdsDataset(basedir=data_dir, split="val", transform=test_transform, img_folder=img_folder)

n_dataPoints = len(valset)
all_idx = [i for i in range(n_dataPoints)]

loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True, "reweight_places": None}
train_loader = get_loader(
    trainset, train=True, reweight_groups=False, reweight_classes=False,
    **loader_kwargs)
test_loader = get_loader(
    testset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)
val_loader = get_loader(
    valset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)
logger.write(f"***main***\n\ttrain_set: {len(trainset)}\n\n")
logger.write(f"\tval_set: {len(valset)}\n\n")
logger.write(f"\ttest_set: {len(testset)}\n\n")

# Load model
n_classes = trainset.n_classes
model = torchvision.models.resnet50(pretrained=False)
d = model.fc.in_features
model.fc = torch.nn.Linear(d, n_classes)
model.load_state_dict(torch.load(args.ckpt_path)) 
model.cuda()
model.eval()

# Evaluate model
logger.write("***main***\nBase Model\n")
base_model_results = {}
get_yp_func = partial(get_y_p, n_places=trainset.n_places)
base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
logger.write(f"***main***Baseline model performance\n")
for i in base_model_results:
    logger.write(f"{i}:\n")
    res = base_model_results[i]
    for k in res:
        logger.write(f"\t{k}:\n")
        logger.write(f"\t\t{res[k]}\n")
logger.write("\n\n\n")

all_results = {}

model.eval()

# Extract embeddings
def get_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x

all_embeddings = {}
all_y, all_p, all_g = {}, {}, {}
for name, loader in [("test", test_loader), ("val", val_loader)]:
    all_embeddings[name] = []
    all_y[name], all_p[name], all_g[name] = [], [], []
    for x, y, g, p in tqdm.tqdm(loader):
        with torch.no_grad():
            all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
            all_y[name].append(y.detach().cpu().numpy())
            all_g[name].append(g.detach().cpu().numpy())
            all_p[name].append(p.detach().cpu().numpy())
    all_embeddings[name] = np.vstack(all_embeddings[name])
    all_y[name] = np.concatenate(all_y[name])
    all_g[name] = np.concatenate(all_g[name])
    all_p[name] = np.concatenate(all_p[name])

def get_results(all_embeddings, all_y, all_g, all_p, logger, current_index, out_dir, flag, balance_val):
    current_results = {}
    c, w1, w2 = dfr_tune(current_index, out_dir, flag=flag, balance_val=balance_val, num_retrains=num_retrains, ratio=ratio)
    current_results["best_hypers"] = (c, w1, w2)
    logger.write(f"***main***\nHypers: {(c, w1, w2)} \n")
    results = dfr_eval(c, w1, w2, all_embeddings, all_y, all_g, all_p, logger, current_index, out_dir, flag=flag, balance_val=balance_val)
    current_results.update({key: results[key] for key in results})
    logger.write(f"***main***\ndfr_val_balance_results***\n{current_results} \n\n")
    return current_results

flag = 'dfr'
balance_val = True
current_index = all_idx
logger.write(f"***main***\n========================={flag}===metaBalance {balance_val}==={len(current_index)}=========================\n")
current_results = get_results(all_embeddings, all_y, all_g, all_p, logger, current_index, out_dir, flag, balance_val, num_retrains=5)
all_results[f"{flag}_metaBalance_{balance_val}"] = current_results

flag = 'slim'
balance_val = False
current_index = valIndex
logger.write(f"***main***\n========================={flag}===metaBalance {balance_val}==={len(current_index)}=========================\n")
current_results = get_results(all_embeddings, all_y, all_g, all_p, logger, current_index, out_dir, flag, balance_val, num_retrains=10, ratio=0.)
all_results[f"{flag}_metaBalance_{balance_val}"] = current_results

logger.write(f"\n\n=============main=============\n=============all_results=============\n==========================\n\n")
for k in all_results:
    res = all_results[k]
    logger.write(f"============={k}=============\n")
    for i in res:
        logger.write(f"{i}:\n")
        logger.write(f"\t{res[i]}\n")

with open(os.path.join(out_dir, "results.pkl"), 'wb') as f:
    pickle.dump(all_results, f)