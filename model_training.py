import sys
sys.path.append("../")

import torch
import torchvision
import numpy as np
import os
import tqdm
import argparse
import sys
import json
from functools import partial
from collections import Counter
from utils.custom_data import CustomDataset, get_loader, get_transform, log_data, getIndexesDataLoader, calc_weights
from utils.utils import Logger, AverageMeter, set_seed, evaluate, get_y_p, update_dict, get_results, get_time
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

workspace_path = 'workspace_path'

def plot_arrays(x_vals, y_vals, x_name, y_name, out_dir):
    temp_name = "{}_vs_{}".format(x_name, y_name)
    plt.figure()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(temp_name)
    plt.plot(x_vals, y_vals)
    plt.savefig(os.path.join(out_dir, temp_name+".png"))
    plt.close()

def argparser():
    parser = argparse.ArgumentParser(description="Train model on waterbirds data")
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--data_dir", type=str, default="waterbird_data_dir")
    parser.add_argument("--exp_info", type=str, default="")
    parser.add_argument("--train_split_path", type=str, default=None, help="Path to the training data subset after SLIM data construction")
    parser.add_argument("--basic_data_aug", action='store_true', help="Train data augmentation")
    parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
    parser.add_argument("--weighted_loss", action='store_true', help="Weighted loss when training data is not balanced")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.)
    return parser

output_dir = "outputs"
def main(args):
    set_seed(args.seed)
    cur_time = get_time()
    target_resolution = (224, 224)
    train_transform = get_transform(target_resolution=target_resolution, train=True, basic_data_aug=args.basic_data_aug)
    test_transform = get_transform(target_resolution=target_resolution, train=False, basic_data_aug=args.basic_data_aug)
    data_dir = args.data_dir
    print(f"load {args.dataset} from {data_dir}")
    if args.dataset == 'waterbirds':
        # Data
        trainset = CustomDataset(basedir=data_dir, split="train", transform=train_transform)
        valset = CustomDataset(basedir=data_dir, split="val", transform=test_transform)
        testset = CustomDataset(basedir=data_dir, split="test", transform=test_transform)
    elif args.dataset == 'celebA':
        # Data
        trainset = CustomDataset(basedir=data_dir, split="train", transform=train_transform, img_folder='img_align_celeba')
        valset = CustomDataset(basedir=data_dir, split="val", transform=test_transform, img_folder='img_align_celeba')
        testset = CustomDataset(basedir=data_dir, split="test", transform=test_transform, img_folder='img_align_celeba')
    n_dataPoints = len(trainset)
    all_idx = [i for i in range(n_dataPoints)]
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    if args.train_split_path is not None:
        if 'annotation' in args.train_split_path:
            output_dir = os.path.join(output_dir, f'{args.exp_info}loadedAnnotatedActiveSet_{len(train_splitIdx)}_{cur_time}')
        else:
            output_dir = os.path.join(output_dir, f'{args.exp_info}loadedActiveSet_{len(train_splitIdx)}_{cur_time}')
        os.makedirs(output_dir, exist_ok=True)
        logger = Logger(os.path.join(output_dir, 'log.txt'))
        logger.write(f"\n\nTraining data: load activeSet from: {args.train_split_path}; ")
        logger.write(f"Training data size: {len(train_splitIdx)} of {n_dataPoints}\n\n")
        train_loader = getIndexesDataLoader(trainset, train_splitIdx, batch_size=args.batch_size)
        train_labels = trainset.y_array[train_splitIdx]
    else:
        train_splitIdx = all_idx
        output_dir = os.path.join(output_dir, f'{args.exp_info}baseline_{cur_time}')
        os.makedirs(output_dir, exist_ok=True)
        logger = Logger(os.path.join(output_dir, 'log.txt'))
        logger.write(f"\n\nTraining data size: {n_dataPoints}\n\n")
        train_loader = get_loader(trainset, train=True, **loader_kwargs)
        train_labels = trainset.y_array[train_splitIdx]
    print('Preparing directory %s' % output_dir)
    with open(os.path.join(output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        args_json = json.dumps(vars(args))
        f.write(args_json)

    if args.weighted_loss:
        temp = Counter(train_labels)
        negative = temp[0]
        positive = temp[1]
        logger.write(f"\n\nTotal instances: {len(train_labels)} | weighted_loss: negative ({negative}), positive ({positive})\n\n")
        weights = calc_weights(negative, positive)
        weights = torch.tensor(weights)
        logger.write(f"\n\nweighted_loss: {weights}\n\n")
        criterion = torch.nn.CrossEntropyLoss(weight=weights.double().float().to(f"cuda:{args.gpu}")).to(f"cuda:{args.gpu}")
    else:
        criterion = torch.nn.CrossEntropyLoss().to(f"cuda:{args.gpu}")

    val_loader = get_loader(valset, train=False, **loader_kwargs)
    test_loader = get_loader(testset, train=False, **loader_kwargs)
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    log_data(logger, trainset, valset, testset, get_yp_func=get_yp_func)
    
    logger.write(f"\n\n\n===============================args====================================\n{args}\n\n\n")
    
    # Model
    n_classes = trainset.n_classes
    model = torchvision.models.resnet50(pretrained=True)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)
    model.to(f"cuda:{args.gpu}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7], gamma=0.5)
    else:
        scheduler = None
    
    logger.flush()
    
    plot_epoch_xvalues = []
    train_loss_list = []
    val_loss_list = []
    best_val_loss = {'epoch': None, 'loss': 100.}
    PATIENCE = 20
    cur_patience = 0
    # Train loop
    for epoch in range(args.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_groups = {g_idx : AverageMeter() for g_idx in range(trainset.n_groups)}
        auroc_f1_groups = {
            'all': {'labels': [], 'preds_prob': [], 'preds_prob_sigmoid': [], 'preds': []},
            'no_patch': {'labels': [], 'preds_prob': [], 'preds_prob_sigmoid': [], 'preds': []}
        }
        for batch in tqdm.tqdm(train_loader):
            x, y, g, p = batch
            x, y, p = x.to(f"cuda:{args.gpu}"), y.to(f"cuda:{args.gpu}"), p.to(f"cuda:{args.gpu}")
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss, x.size(0))
            preds_prob = torch.softmax(logits, dim=1)[:, 1] # binary classification
            preds_prob_sigmoid = torch.sigmoid(logits)[:, 1]
            update_dict(acc_groups, auroc_f1_groups, y, g, p, logits, preds_prob, preds_prob_sigmoid)
        if args.scheduler:
            scheduler.step()
        epoch_loss = loss_meter.avg.detach().cpu().numpy()
        logger.write(f"Epoch {epoch}\t Loss: {epoch_loss}\n")
        
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
        results = get_results(acc_groups, all_f1, all_auroc, all_auroc_sigmoid, all_accuracy, no_patch_f1, no_patch_auroc, no_patch_auroc_sigmoid, no_patch_accuracy, get_yp_func, epoch_loss)
        
        
        logger.write(f"Train results \n")
        for k in results:
            logger.write(f"\t{k}: {results[k]}\n")
        plot_epoch_xvalues.append(epoch+1)
        train_loss_list.append(epoch_loss)
        
        results = evaluate(model, val_loader, get_yp_func, args)
        logger.write("Val results \n")
        for k in results:
            logger.write(f"\t{k}: {results[k]}\n")
        loss = results["loss"]
        val_loss_list.append(loss)
        if loss < best_val_loss['loss']:
            best_val_loss.update({key: results[key] for key in results})
            best_val_loss['epoch'] = epoch
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model', f'epoch_{epoch}_checkpoint.pt'))
        
        logger.flush()
        if cur_patience > PATIENCE:
            logger.write(f"\n\n\ncur_patience: {cur_patience} | break\n\n\n")
            break
    results = evaluate(model, test_loader, get_yp_func, args)
    logger.write("Last Epoch Test results \n")
    for k in results:
        logger.write(f"\t{k}: {results[k]}\n")
    
    logger.write(f"\nBest loss (val): \n")
    for k in best_val_loss:
        logger.write(f"\t{k}: {best_val_loss[k]}\n")

    plot_arrays(plot_epoch_xvalues, train_loss_list, 'Epochs', 'TrainLoss', output_dir)
    plot_arrays(plot_epoch_xvalues, val_loss_list, 'Epochs', 'ValLoss', output_dir)


if __name__ == "__main__":
    args = argparser().parse_args()
    main(args)