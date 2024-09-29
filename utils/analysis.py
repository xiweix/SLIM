import faiss
import numpy as np
from copy import deepcopy
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from datetime import datetime
import os
from PIL import Image
import tqdm
import random
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import umap
from utils.gradcam import GradCAM
from utils.evaluation import get_att_eval
from utils.plot import img_for_annotation, red_orange_blue
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
from sklearn.preprocessing import normalize

def get_time():
    # Get the current time with microsecond precision
    now = datetime.now()
    # Format the string to include year, month, day, hour, minute, second, and microsecond
    time_str = f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}_{now.microsecond:04d}"
    return time_str

def faiss_k_means(data, ncentroids, gpu=True, niter=100, verbose=False, seed=0, nredo=3):
    kmeans = faiss.Kmeans(data.shape[1], ncentroids, gpu=gpu, niter=niter, verbose=verbose, seed=seed, nredo=nredo)
    kmeans.train(data)
    D, I = kmeans.index.search(data, 1)
    return kmeans, I[:, 0]

def evaluate_clustering(data, labels):
    # Only evaluate clusters with more than one cluster and not all points as noise
    if len(set(labels)) > 1 and not (len(set(labels)) == 1 and -1 in labels):
        return silhouette_score(data, labels)
    else:
        return None

def plot_silhouette_scores(silhouette_scores):
    sizes, scores = zip(*silhouette_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, scores, marker='o')
    plt.xlabel('Min Cluster Size')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different min_cluster_sizes')
    plt.show()

def plot_clusters(data, labels):
    plt.figure(figsize=(15, 7))
    # Generate a list of unique labels and colors
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    # Plot each cluster with a different color and label
    for label in unique_labels:
        clustered_data = data[labels == label]
        plt.scatter(clustered_data[:, 0], clustered_data[:, 1], color=colors(label), label=f'Cluster {label}', s=6)
    plt.title('Scatter Plot of Embeddings Colored by Cluster Labels')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.legend(title='Clusters')
    plt.show()

def new_norm(x, CORE=None, SPUR=None):
    """
    Map median of x to 0, and normalize [0, max] & [min, 0] to [0, 1] & [-1, 0], respectively
    """
    x_norm = x - np.median(x)
    x_copy = deepcopy(x_norm)
    x_min = np.min(x_norm)
    x_max = np.max(x_norm)
    x_a = (x_norm - x_min) / (0 - x_min) - 1
    x_b = x_norm / x_max
    x_norm[x_copy<0] = x_a[x_copy<0]
    x_norm[x_copy>0] = x_b[x_copy>0]
    if CORE is not None:
        c = CORE - np.median(x)
        s = SPUR - np.median(x)
        if c < 0:
            c = (c - x_min) / (0 - x_min) - 1
        else:
            c = c / x_max
        if s < 0:
            s = (s - x_min) / (0 - x_min) - 1
        else:
            s = s / x_max
        return x_norm, c, s
    return x_norm

def propagate_core_spurious(data, labels, method="LabelSpreading", n_neighbors=30, gamma=0.0005, alpha=0.2, max_iter=100000):
    # Preprocessing
    data = (data-data.mean())/data.std()
    n_labels_positive = len([x for x in labels if x == 1])
    n_labels_negative = len([x for x in labels if x == 0])  
    # Create additional labels if necessary
    if n_labels_positive == 0 and n_labels_negative == 0:
        raise ValueError("No labels given to clusters. Labels should be +1 (core) or -1 (spurious)")
    if n_labels_positive == 0 or n_labels_negative == 0:
        given_labels = 1 if n_labels_negative == 0 else 0
        print("n_labels_positive: ", n_labels_positive, "n_labels_negative: ", n_labels_negative, "given_labels: ", given_labels)
        core_spur_score = np.ones(labels.shape)
        return core_spur_score
    # Propagate labels with LabelPropagation
    if method == 'LabelSpreading':
        lp = LabelSpreading(kernel='rbf', gamma=gamma, alpha=alpha, max_iter=max_iter)
    else:
        lp = LabelPropagation(kernel='knn', n_neighbors=n_neighbors, max_iter=max_iter)
    if len(labels) > 50000:
        print(f"data size: {len(data)}. Apply downsampling to fit {method}.")
        # Separate labeled and unlabeled data
        labeled_indices = np.where((labels == 0) | (labels == 1))[0]
        unlabeled_indices = np.where(labels == -1)[0]
        # Sample from the unlabeled data
        # sample_size = int(0.3 * len(unlabeled_indices))  # Adjust as necessary
        sample_size = 50000  # Adjust as necessary
        sampled_unlabeled_indices = np.random.choice(unlabeled_indices, sample_size, replace=False)
        # Combine labeled and sampled unlabeled data
        subset_indices = np.concatenate((labeled_indices, sampled_unlabeled_indices))
        print(f"Downsample {len(data)} into {len(subset_indices)} ({sample_size} + {len(labeled_indices)}).")
        subset_data = data[subset_indices]
        subset_labels = labels[subset_indices]
        # Fit Label Spreading model on the subset
        lp.fit(subset_data, subset_labels)
    else:
        print(f"data size: {len(data)}. Fit {method}.")
        lp.fit(data, labels)
    probas = lp.predict_proba(data)
    probas = probas / np.atleast_2d(probas.sum(axis=1)).T
    core_spur_score = probas[:, 1]
    core_spur_score = (core_spur_score-core_spur_score.min())/(core_spur_score.max() - core_spur_score.min())  ## to be in range 0 and 1
    return core_spur_score

def get_attention_new(model, loader, data_dir, out_dir, data_flag='val', dataset_name='waterbirds', model_type='resnet50', layer_name='layer4', device='cuda:0', seed_value=1):
    random.seed(seed_value)
    np.random.seed(seed_value)
    ### get gradcam at first
    gradcam_dir_2 = os.path.join(out_dir, 'gradCAM_reverse')
    last_img_name = loader.dataset[-1][5]
    if not os.path.exists(os.path.join(gradcam_dir_2, f'{last_img_name}_gradcam_img_contour.jpg')):
        model.eval()
        model_dict = dict(type=model_type, arch=model, layer_name=layer_name, input_size=(224, 224), device=torch.device(device))
        gradcam = GradCAM(model_dict, True)
        gradcam_dir = os.path.join(out_dir, 'gradCAM')
        os.makedirs(gradcam_dir, exist_ok=True)
        os.makedirs(gradcam_dir_2, exist_ok=True)
        val_index = []
        total_entropy_scores = []
        total_attention_scores = []
        total_img_names = []
        total_img_paths = []
        total_label = []
        total_prediction = []
        opposite_attention_scores = []
        for i, (idx, cur_batch, label, g, p, img_name, img_path) in enumerate(tqdm.tqdm(loader, desc="Extracting Representations, attention, etc.")):
            val_index.append(int(idx[0]))
            img_name = img_name[0]
            total_img_names.append(img_name)
            img_path = img_path[0]
            total_img_paths.append(img_path)
            label = int(label[0])
            total_label.append(label)
            cur_batch = cur_batch.cuda(device)
            pil_img = Image.open(img_path).convert('RGB')
            w, h = pil_img.size
            img_sized_mask, logits, _, small_mask, _, activations = gradcam(cur_batch, class_idx=label, out_size=(h, w))   ### use predicted class to generate gradcam         
            ### attention map
            img_sized_mask = np.nan_to_num(img_sized_mask.squeeze().cpu().numpy())
            copied_img_sized_mask = deepcopy(img_sized_mask)
            np.save(os.path.join(gradcam_dir, f'{img_name}_gradcam_mask.npy'), img_sized_mask)
            copied_small_mask = small_mask.detach().clone()
            # ### add this to normalize:
            # small_mask = (small_mask - small_mask.min()).div(small_mask.max() - small_mask.min())
            small_mask = small_mask.detach().squeeze().cpu().numpy()
            small_mask = np.nan_to_num(small_mask)
            ### feature vectors
            vector = activations.detach().squeeze().cpu().numpy()
            vector_mask_temp = np.expand_dims(small_mask, axis=0)
            vector_mask_expand = np.repeat(vector_mask_temp, vector.shape[0], axis=0)
            res = np.sum(vector * vector_mask_expand, axis=(1, 2)) ### weighted sum
            np.save(os.path.join(gradcam_dir, f'{img_name}_feature_vector.npy'), vector)
            np.save(os.path.join(gradcam_dir, f'{img_name}_weighted_feature_vector.npy'), res)
            np.save(os.path.join(gradcam_dir, f'{img_name}_small_mask.npy'), small_mask)
            ### entropy_score
            outputs = torch.nn.functional.softmax(logits, dim=1)
            entropy = outputs * torch.log2(outputs)
            entropy = -1*torch.sum(entropy, dim=1).detach().cpu().numpy()[0]
            total_entropy_scores.append(entropy)
            pred = outputs.argmax(dim=1, keepdim=True)
            pred = pred.flatten().tolist()
            total_prediction += pred
            if dataset_name == 'waterbirds':
                ### get_att_eval according to GT segmentation mask 
                segment_dir = data_dir.replace('waterbird_complete95_forest2water2', 'segmentation_masks')
                seg = np.load(os.path.join(segment_dir, img_name.replace('jpg', 'npy')))
                att_score = get_att_eval(img_sized_mask, gt_mask=seg)
            else:
                att_score = 0.
            total_attention_scores.append(att_score)
            ### save gradCAM
            heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_img_sized_mask) * 255))
            heatmapWimg = deepcopy(pil_img)
            heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_img_sized_mask.shape)*0.5*255)))
            heatmap = heatmap.convert('RGB')
            heatmapWimg = heatmapWimg.convert('RGB')
            heatmap.save(os.path.join(gradcam_dir, f'{img_name}_gradcam.jpg'))
            heatmapWimg.save(os.path.join(gradcam_dir, f'{img_name}_gradcam_img.jpg'))
            copied_img_sized_mask[copied_img_sized_mask<0.8] = 0.0
            heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_img_sized_mask) * 255))
            heatmapWimg = deepcopy(pil_img)
            heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_img_sized_mask.shape)*0.5*255)))
            heatmapWimg = heatmapWimg.convert('RGB')
            heatmapWimg.save(os.path.join(gradcam_dir, f'{img_name}_gradcam_img_contour.jpg'))
            copied_small_mask = (copied_small_mask - copied_small_mask.min()).div(copied_small_mask.max() - copied_small_mask.min())
            opposite_small_mask = 1 - copied_small_mask
            opposite_img_sized_mask = F.upsample(opposite_small_mask, size=(h, w), mode='bilinear', align_corners=False)
            opposite_img_sized_mask_min, opposite_img_sized_mask_max = opposite_img_sized_mask.min(), opposite_img_sized_mask.max()
            opposite_img_sized_mask = (opposite_img_sized_mask - opposite_img_sized_mask_min).div(opposite_img_sized_mask_max - opposite_img_sized_mask_min).data
            opposite_img_sized_mask = np.nan_to_num(opposite_img_sized_mask.squeeze().cpu().numpy())
            copied_opposite_img_sized_mask = deepcopy(opposite_img_sized_mask)
            np.save(os.path.join(gradcam_dir_2, f'{img_name}_gradcam_mask.npy'), opposite_img_sized_mask)
            opposite_small_mask = opposite_small_mask.detach().squeeze().cpu().numpy()
            opposite_small_mask = np.nan_to_num(opposite_small_mask)
            opposite_vector_mask_temp = np.expand_dims(opposite_small_mask, axis=0)
            opposite_vector_mask_expand = np.repeat(opposite_vector_mask_temp, vector.shape[0], axis=0)
            opposite_res = np.sum(vector * opposite_vector_mask_expand, axis=(1, 2)) ### weighted sum
            np.save(os.path.join(gradcam_dir, f'{img_name}_reverse_weighted_feature_vector.npy'), opposite_res)
            np.save(os.path.join(gradcam_dir, f'{img_name}_reverse_small_mask.npy'), opposite_small_mask)
            if dataset_name == 'waterbirds':
                opposite_att_score = get_att_eval(opposite_img_sized_mask, gt_mask=seg)
            else:
                opposite_att_score = 0.
            opposite_attention_scores.append(opposite_att_score)
            ### save 1-gradCAM
            heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_opposite_img_sized_mask) * 255))
            heatmapWimg = deepcopy(pil_img)
            heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_opposite_img_sized_mask.shape)*0.5*255)))
            heatmap = heatmap.convert('RGB')
            heatmapWimg = heatmapWimg.convert('RGB')
            heatmap.save(os.path.join(gradcam_dir_2, f'{img_name}_gradcam.jpg'))
            heatmapWimg.save(os.path.join(gradcam_dir_2, f'{img_name}_gradcam_img.jpg'))
            copied_opposite_img_sized_mask[copied_opposite_img_sized_mask<0.8] = 0.0
            heatmap = Image.fromarray(np.uint8(matplotlib.cm.jet(copied_opposite_img_sized_mask) * 255))
            heatmapWimg = deepcopy(pil_img)
            heatmapWimg.paste(heatmap, (0, 0), mask=Image.fromarray(np.uint8(np.ones(copied_opposite_img_sized_mask.shape)*0.5*255)))
            heatmapWimg = heatmapWimg.convert('RGB')
            heatmapWimg.save(os.path.join(gradcam_dir_2, f'{img_name}_gradcam_img_contour.jpg'))
        total_entropy_scores = np.array(total_entropy_scores)
        total_attention_scores = np.array(total_attention_scores)
        opposite_attention_scores = np.array(opposite_attention_scores)
        np.save(os.path.join(out_dir, f'{data_flag}_total_entropy.npy'), total_entropy_scores)
        np.save(os.path.join(out_dir, f'{data_flag}_total_attention.npy'), total_attention_scores)
        np.save(os.path.join(out_dir, f'{data_flag}_opposite_attention.npy'), opposite_attention_scores)
        np.save(os.path.join(out_dir, f'{data_flag}_total_label.npy'), total_label)
        np.save(os.path.join(out_dir, f'{data_flag}_total_prediction.npy'), total_prediction)
        np.save(os.path.join(out_dir, f'{data_flag}_total_img_names.npy'), total_img_names)
        np.save(os.path.join(out_dir, f'{data_flag}_total_img_paths.npy'), total_img_paths)
        loader.dataset.no_aug = False
        print(total_entropy_scores.shape, total_attention_scores.shape)
    else:
        total_entropy_scores = np.load(os.path.join(out_dir, f'{data_flag}_total_entropy.npy'))
        total_attention_scores = np.load(os.path.join(out_dir, f'{data_flag}_total_attention.npy'))
        total_label = np.load(os.path.join(out_dir, f'{data_flag}_total_label.npy'))
        total_prediction = np.load(os.path.join(out_dir, f'{data_flag}_total_prediction.npy'))
        total_img_names = np.load(os.path.join(out_dir, f'{data_flag}_total_img_names.npy'))
        total_img_paths = np.load(os.path.join(out_dir, f'{data_flag}_total_img_paths.npy'))
    if not os.path.exists(os.path.join(out_dir, f'{data_flag}_opposite_weighted_feature_vector.npy')):
        total_weighted_vectors = []
        total_cam_masks = []
        opposite_weighted_vectors = []
        opposite_cam_masks = []
        gradcam_dir = os.path.join(out_dir, 'gradCAM')
        gradcam_dir_2 = os.path.join(out_dir, 'gradCAM_reverse')
        for img_name in total_img_names:
            res = np.load(os.path.join(gradcam_dir, f'{img_name}_weighted_feature_vector.npy'))
            small_mask = np.load(os.path.join(gradcam_dir, f'{img_name}_small_mask.npy'))
            total_weighted_vectors.append(res)
            total_cam_masks.append(small_mask)
            opposite_res = np.load(os.path.join(gradcam_dir, f'{img_name}_reverse_weighted_feature_vector.npy'))
            opposite_small_mask = np.load(os.path.join(gradcam_dir, f'{img_name}_reverse_small_mask.npy'))
            opposite_weighted_vectors.append(opposite_res)
            opposite_cam_masks.append(opposite_small_mask)
        total_weighted_vectors = np.array(total_weighted_vectors)
        total_masks_flatten = np.array(total_cam_masks)
        total_masks_flatten = total_masks_flatten.reshape((total_masks_flatten.shape[0], -1))
        opposite_weighted_vectors = np.array(opposite_weighted_vectors)
        opposite_masks_flatten = np.array(opposite_cam_masks)
        opposite_masks_flatten = opposite_masks_flatten.reshape((opposite_masks_flatten.shape[0], -1))
        np.save(os.path.join(out_dir, f'{data_flag}_total_small_mask.npy'), total_cam_masks)
        np.save(os.path.join(out_dir, f'{data_flag}_total_small_mask_flatten.npy'), total_masks_flatten)
        np.save(os.path.join(out_dir, f'{data_flag}_total_weighted_feature_vector.npy'), total_weighted_vectors)
        np.save(os.path.join(out_dir, f'{data_flag}_opposite_small_mask.npy'), opposite_cam_masks)
        np.save(os.path.join(out_dir, f'{data_flag}_opposite_small_mask_flatten.npy'), opposite_masks_flatten)
        np.save(os.path.join(out_dir, f'{data_flag}_opposite_weighted_feature_vector.npy'), opposite_weighted_vectors)
    else:
        total_cam_masks = np.load(os.path.join(out_dir, f'{data_flag}_total_small_mask.npy'))
        total_masks_flatten = np.load(os.path.join(out_dir, f'{data_flag}_total_small_mask_flatten.npy'))
        total_weighted_vectors = np.load(os.path.join(out_dir, f'{data_flag}_total_weighted_feature_vector.npy'))
        opposite_weighted_vectors = np.load(os.path.join(out_dir, f'{data_flag}_opposite_weighted_feature_vector.npy'))
        opposite_masks_flatten = np.load(os.path.join(out_dir, f'{data_flag}_opposite_small_mask_flatten.npy'))
    if not os.path.exists(os.path.join(out_dir, f'{data_flag}_supervised_opposite_weighted_fv_embedding.npy')):
        if len(total_entropy_scores) > 100000:
            reducer = umap.UMAP(
                n_neighbors=3, ###5
                min_dist=0.000,
                n_components=2, ### can be larger for slicefinding
                random_state=0,
                spread=1.9,
                metric='euclidean',
            )
            sample_size = 5000
            random_indices = np.random.choice(len(total_weighted_vectors), sample_size, replace=False)
            reducer.fit(total_weighted_vectors[random_indices])
            total_w_fv_embeddings = reducer.transform(total_weighted_vectors)
            plt.figure(figsize=(15, 7))
            plt.scatter(total_w_fv_embeddings[:, 0], total_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_total_att_space_embedding.npy'), total_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_total_weighted_fv_embedding.npy'), total_w_fv_embeddings)
            reducer.fit(total_weighted_vectors[random_indices], y=total_label[random_indices])
            supervised_w_fv_embeddings = reducer.transform(total_weighted_vectors)
            plt.figure(figsize=(15, 7))
            plt.scatter(supervised_w_fv_embeddings[:, 0], supervised_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_att_space_embedding.npy'), supervised_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_weighted_fv_embedding.npy'), supervised_w_fv_embeddings)
            sample_size = 5000
            random_indices = np.random.choice(len(opposite_weighted_vectors), sample_size, replace=False)
            reducer.fit(opposite_weighted_vectors[random_indices])
            opposite_w_fv_embeddings = reducer.transform(opposite_weighted_vectors)
            # opposite_w_fv_embeddings = normalize(opposite_w_fv_embeddings)
            plt.figure(figsize=(15, 7))
            plt.scatter(opposite_w_fv_embeddings[:, 0], opposite_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_opposite_att_space_embedding.npy'), opposite_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_opposite_weighted_fv_embedding.npy'), opposite_w_fv_embeddings)
            reducer.fit(opposite_weighted_vectors[random_indices], y=total_label[random_indices])
            supervised_opposite_w_fv_embeddings = reducer.transform(opposite_weighted_vectors)
            plt.figure(figsize=(15, 7))
            plt.scatter(supervised_opposite_w_fv_embeddings[:, 0], supervised_opposite_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_opposite_att_space_embedding.npy'), supervised_opposite_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_opposite_weighted_fv_embedding.npy'), supervised_opposite_w_fv_embeddings)
        else:
            reducer = umap.UMAP(
                n_neighbors=5, ###5
                min_dist=0.000,
                n_components=2, ### can be larger for slicefinding
                random_state=0,
                metric='euclidean',
            )
            reducer.fit(total_weighted_vectors)
            total_w_fv_embeddings = reducer.transform(total_weighted_vectors)
            plt.figure(figsize=(15, 7))
            plt.scatter(total_w_fv_embeddings[:, 0], total_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_total_att_space_embedding.npy'), total_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_total_weighted_fv_embedding.npy'), total_w_fv_embeddings)
            reducer.fit(total_weighted_vectors, y=total_label)
            supervised_w_fv_embeddings = reducer.transform(total_weighted_vectors)
            plt.figure(figsize=(15, 7))
            plt.scatter(supervised_w_fv_embeddings[:, 0], supervised_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_att_space_embedding.npy'), supervised_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_weighted_fv_embedding.npy'), supervised_w_fv_embeddings)
            reducer.fit(opposite_weighted_vectors)
            opposite_w_fv_embeddings = reducer.transform(opposite_weighted_vectors)
            # opposite_w_fv_embeddings = normalize(opposite_w_fv_embeddings)
            plt.figure(figsize=(15, 7))
            plt.scatter(opposite_w_fv_embeddings[:, 0], opposite_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_opposite_att_space_embedding.npy'), opposite_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_opposite_weighted_fv_embedding.npy'), opposite_w_fv_embeddings)
            reducer.fit(opposite_weighted_vectors, y=total_label)
            supervised_opposite_w_fv_embeddings = reducer.transform(opposite_weighted_vectors)
            # supervised_opposite_w_fv_embeddings = normalize(supervised_opposite_w_fv_embeddings)
            plt.figure(figsize=(15, 7))
            plt.scatter(supervised_opposite_w_fv_embeddings[:, 0], supervised_opposite_w_fv_embeddings[:, 1], s=6)
            plt.show()
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_opposite_att_space_embedding.npy'), supervised_opposite_w_fv_embeddings)
            np.save(os.path.join(out_dir, f'{data_flag}_supervised_opposite_weighted_fv_embedding.npy'), supervised_opposite_w_fv_embeddings)
    else:
        total_w_fv_embeddings = np.load(os.path.join(out_dir, f'{data_flag}_total_att_space_embedding.npy'))
        total_w_fv_embeddings = np.load(os.path.join(out_dir, f'{data_flag}_total_weighted_fv_embedding.npy'))
        total_weighted_vectors = np.load(os.path.join(out_dir, f'{data_flag}_total_weighted_feature_vector.npy'))
        opposite_w_fv_embeddings = np.load(os.path.join(out_dir, f'{data_flag}_opposite_weighted_fv_embedding.npy'))
    return total_attention_scores, total_entropy_scores, total_prediction, total_label, total_img_names, total_img_paths, total_w_fv_embeddings, total_cam_masks, total_weighted_vectors, opposite_weighted_vectors

def attention_annotation_new(auto_att_scores, total_img_names, total_labels, total_prediction, total_img_paths, att_rep_embeddings, read_dir, flag, seed=0, att_budget_dict={0: 40, 1: 30}, gamma=[35, 35]):
    np.random.seed(seed)
    out_dir = os.path.join(read_dir, f'att_budget_{att_budget_dict[0]}_{att_budget_dict[1]}_{flag}')
    print("att_budget_dict: \n\t", att_budget_dict)
    os.makedirs(out_dir, exist_ok=True)
    # shutil.copyfile('/home/dx/Projects/hci_active_learning/code/deep-active-learning-pytorch/tools/spur_mitigate_sampling.py', oj(out_dir, 'spur_mitigate_sampling.py'))
    unique_labels = np.unique(total_labels)
    if not os.path.exists(os.path.join(out_dir, 'total_propagated_attention_scores.npy')):
        total_att_scores = np.ones(len(att_rep_embeddings)) * -1
        for specified_label in unique_labels:
            n_clusters_per_label = att_budget_dict[specified_label]
            att_budget_per_label = att_budget_dict[specified_label]
            cur_all_index = np.where(total_labels == specified_label)[0]
            print("specified_label: ", specified_label, "cur_all_index: ", cur_all_index.shape)
            cur_label_att_scores = np.ones(len(cur_all_index), dtype=int) * -1
            print("\n\nCreate cur_label_att_scores to save attention annotation: ", cur_label_att_scores.shape)
            print(f"Annotation budget: {att_budget_per_label} ({att_budget_per_label}/{len(cur_all_index)} = {att_budget_per_label / (len(cur_all_index)) * 100}%)")
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_cluster_labels.npy"), cur_all_index)
            kmeans_weighted, cluster_labels = faiss_k_means(att_rep_embeddings[cur_all_index], n_clusters_per_label, seed=seed)
            centroids = kmeans_weighted.centroids
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_cluster_labels.npy"), cluster_labels)
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_centroids.npy"), centroids)
            data_float32 = att_rep_embeddings[cur_all_index].astype(np.float32)
            # Create a FAISS index for the data points
            d = data_float32.shape[1]  # dimension of the vectors
            index = faiss.IndexFlatL2(d)   # Using L2 distance
            index.add(data_float32)        # Add data points to the index
            centroids_float32 = centroids.astype(np.float32)
            top_n = 5
            D, I = index.search(centroids_float32, top_n)
            closest_points_idx = np.empty(centroids_float32.shape[0], dtype=int)
            for i in range(centroids_float32.shape[0]):
                # Randomly select one point from the top_n for each centroid. If using the top closest point, use I[i, 0] instead of random.choice(I[i])
                closest_points_idx[i] = random.choice(I[i])
            closest_points_embeddings = att_rep_embeddings[cur_all_index][closest_points_idx]
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_centroids_closest_points_embeddings.npy"), closest_points_embeddings)
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_centroids_closest_points_idx.npy"), closest_points_idx)
            plt.figure(figsize=(10, 5))
            plt.scatter(att_rep_embeddings[cur_all_index, 0], att_rep_embeddings[cur_all_index, 1], color='gray', s=1)
            plt.scatter(centroids[:, 0], centroids[:, 1], color='b', s=7)
            plt.scatter(closest_points_embeddings[:, 0], closest_points_embeddings[:, 1], marker='x', color='r')
            plt.savefig(os.path.join(out_dir, f"label_{specified_label}_kmeans_cluster_centroids.jpg"))
            if att_budget_per_label == n_clusters_per_label:
                sampled_clusters = list(range(n_clusters_per_label)) ### keep all 50
            else:
                sampled_clusters = random.sample(list(range(n_clusters_per_label)), att_budget_per_label) ### sample 20
            sampled_closest_points_idx = closest_points_idx[np.array(sampled_clusters)]
            sampled_closest_points_embeddings = att_rep_embeddings[cur_all_index][sampled_closest_points_idx]
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_centroids_sampled_closest_points_embeddings.npy"), sampled_closest_points_embeddings)
            plt.figure(figsize=(10, 5))
            plt.scatter(att_rep_embeddings[cur_all_index, 0], att_rep_embeddings[cur_all_index, 1], color='gray', s=1)
            plt.scatter(centroids[:, 0], centroids[:, 1], color='b', s=7)
            plt.scatter(closest_points_embeddings[:, 0], closest_points_embeddings[:, 1], marker='x', color='b')
            plt.scatter(sampled_closest_points_embeddings[:, 0], sampled_closest_points_embeddings[:, 1], marker='x', color='r')
            plt.savefig(os.path.join(out_dir, f"label_{specified_label}_kmeans_cluster_centroids_sampled.jpg"))
            np.save(os.path.join(out_dir, f"label_{specified_label}_kmeans_sampled_clusters.npy"), sampled_clusters)
            center_cam_path_list = []
            center_cam_contour_path_list = []
            center_img_path_list = []
            title_1 = []
            title_2 = []
            title_3 = []
            print("Read GradCAM images from: ", read_dir, "sampled_clusters: ", len(sampled_clusters))
            for i, cur_cluster in enumerate(sampled_clusters):
                temp_center = closest_points_idx[cur_cluster]
                center = cur_all_index[temp_center]
                center_path_1 = os.path.join(read_dir, 'gradCAM', f'{total_img_names[center]}_gradcam_img.jpg')
                center_path_2 = os.path.join(read_dir, 'gradCAM', f'{total_img_names[center]}_gradcam_img_contour.jpg')
                center_cam_path_list.append(center_path_1)
                center_cam_contour_path_list.append(center_path_2)
                center_img_path_list.append(total_img_paths[center])
                title_1.append(str(i))
                title_2.append(f"pred {total_prediction[center]}")
                title_3.append(f"{auto_att_scores[center]:.4f}")
            att_query_img_path_1 = os.path.join(out_dir, f'att_query_imgs_label_{specified_label}.jpg')
            att_query_img_path_2 = os.path.join(out_dir, f'att_query_space_label_{specified_label}.jpg')
            print("n_sampled_iamges for annotation: ", len(center_img_path_list), len(center_cam_path_list), len(center_cam_contour_path_list))
            img_for_annotation(center_img_path_list, center_cam_path_list, center_cam_contour_path_list, att_rep_embeddings, sampled_closest_points_embeddings, img_title_list_1=title_1, img_title_list_2=title_2, img_title_list_3=title_3, save_path_1=att_query_img_path_1, save_path_2=att_query_img_path_2, num_per_row=15)
            print(f"Image grid (total: {len(center_img_path_list)}) that require attention annotation is saved at: \n\t{att_query_img_path_1}")
            wrong = 1
            while (wrong==1):
                annotation_str = input('Annotate the attention correctness of each image in order (0: wrong; 1: correct; -1: not sure). Please separate by \',\'\n')
                if len(annotation_str) == 0:
                    print(f"Error: No Input. {len(annotation_str)}")
                    wrong = 1
                else:
                    annotation_input = annotation_str.split(',')
                    print(f"\nReceived {len(annotation_input)} input: {annotation_input}.\n{Counter(annotation_input)}.\nStart processing.\n")
                    if len(annotation_input) != len(center_img_path_list):
                        print(f"Error: Input includes {len(annotation_input)} elements.")
                        wrong = 1
                    else:
                        cur_att_annotation = []
                        for a in annotation_input:
                            a = a.strip()
                            a = int(a)
                            if a == 0 or a == 1 or a == -1:
                                wrong = 0
                                cur_att_annotation.append(a)
                            else:
                                wrong = 1
                                print(f"Error: {a} does not belong to {-1, 0, 1}")
                                break
                if wrong == 0:
                    if len(cur_att_annotation) != len(center_img_path_list):
                        print(f"Error: Input includes {len(cur_att_annotation)} elements.")
                        wrong = 1
                if wrong == 1:
                    print(f'The initial input {annotation_str} is wrong. Please redo.')
            np.save(os.path.join(out_dir, f'label_{specified_label}_kmeans_sampled_clusters_att_labels.npy'), cur_att_annotation)
            for i, a in enumerate(cur_att_annotation):
                temp_idx = sampled_closest_points_idx[i]
                # idx = cur_all_index[temp_idx]
                # att_scores[idx] = a
                cur_label_att_scores[temp_idx] = a
            np.save(os.path.join(out_dir, f'label_{specified_label}_annotated_attention_scores.npy'), cur_label_att_scores)
            print(f"specified_label: {specified_label}.\nAnnotated attention collected.\n\tcur_label_att_scores: {len(cur_label_att_scores)}. {Counter(cur_label_att_scores)}")
            print(f"Propagate {len(cur_label_att_scores)} on the att_rep_embeddings: {att_rep_embeddings[cur_all_index].shape}")
            print("att_rep_embeddings includes nan: ", np.isnan(att_rep_embeddings[cur_all_index]).any())
            print("cur_label_att_scores includes nan: ", np.isnan(cur_label_att_scores).any())
            propagated_att_scores = propagate_core_spurious(att_rep_embeddings[cur_all_index], cur_label_att_scores, gamma=gamma[specified_label], alpha=0.000001)
            print("gamma: ", gamma[specified_label])
            print("propagated_att_scores includes nan: ", np.isnan(propagated_att_scores).any())
            np.save(os.path.join(out_dir, f'label_{specified_label}_propagated_attention_scores.npy'), propagated_att_scores)
            print("\n\nPropagated attention scores: ", propagated_att_scores.shape, "cur_all_index: ", cur_all_index.shape)
            print(np.percentile(propagated_att_scores, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            print("propagated_att_scores\t==-1", np.where(propagated_att_scores==-1)[0].shape)
            print("propagated_att_scores\t==0", np.where(propagated_att_scores==0)[0].shape)
            print("propagated_att_scores\t==1", np.where(propagated_att_scores==1)[0].shape)
            print("\n\nBefore assigning propagated labels to total: ")
            print("\t==-1", np.where(total_att_scores==-1)[0].shape)
            print("\t==0", np.where(total_att_scores==0)[0].shape)
            print("\t==1", np.where(total_att_scores==1)[0].shape)
            for i, temp_idx in enumerate(cur_all_index):
                total_att_scores[temp_idx] = propagated_att_scores[i]
            print("After assigning propagated labels to total: ")
            print("\t==-1", np.where(total_att_scores==-1)[0].shape)
            print("\t==0", np.where(total_att_scores==0)[0].shape)
            print("\t==1", np.where(total_att_scores==1)[0].shape)
            plt.figure(figsize=(10, 5))
            plt.scatter(att_rep_embeddings[cur_all_index, 0], att_rep_embeddings[cur_all_index, 1], c=propagated_att_scores, s=.5, cmap=red_orange_blue)
            plt.colorbar()
            plt.scatter(sampled_closest_points_embeddings[:, 0], sampled_closest_points_embeddings[:, 1], marker='x', color='black')
            plt.savefig(os.path.join(out_dir, f"label_{specified_label}_kmeans_cluster_att_score_propagated.jpg"))
        plt.figure(figsize=(10, 5))
        plt.scatter(att_rep_embeddings[:, 0], att_rep_embeddings[:, 1], c=total_att_scores, s=.5, cmap=red_orange_blue)
        plt.colorbar()
        for specified_label in unique_labels:
            sampled_closest_points_embeddings = np.load(os.path.join(out_dir, f"label_{specified_label}_kmeans_centroids_sampled_closest_points_embeddings.npy"))
            plt.scatter(sampled_closest_points_embeddings[:, 0], sampled_closest_points_embeddings[:, 1], marker='x', color='black')
        plt.savefig(os.path.join(out_dir, "kmeans_cluster_att_score_propagated.jpg"))
        np.save(os.path.join(out_dir, 'total_propagated_attention_scores.npy'), total_att_scores)
    else:
        total_att_scores = np.load(os.path.join(out_dir, 'total_propagated_attention_scores.npy'))
    return total_att_scores

def sample_from_group(group_index, budget, group_vector, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    print(f"diversity assisted sampling from group of length {len(group_index)}")
    if budget <= len(group_index):
        kmeans_m, _ = faiss_k_means(group_vector, budget, seed=0)
        centroids = kmeans_m.centroids
        distances = np.linalg.norm(group_vector[:, np.newaxis] - centroids, axis=2)
        top_n = 3
        top_n_indices = np.argsort(distances, axis=0)[:top_n, :]
        closest_points_idx = np.empty(centroids.shape[0], dtype=int)
        for i in range(centroids.shape[0]):
            closest_points_idx[i] = random.choice(top_n_indices[:, i]) 
        return group_index[closest_points_idx]
    sampled_g = group_index
    additional_samples = np.array(random.choices(list(group_index), k=budget-len(group_index)))
    sampled_g = np.concatenate([group_index, additional_samples])
    random.shuffle(sampled_g)
    return sampled_g

def find_elbow_point(SSE):
    # Calculate the first and second derivative of the SSE
    SSE_d1 = np.diff(SSE) / 2  # First derivative
    SSE_d2 = np.diff(SSE_d1) / 2  # Second derivative

    # The elbow point is where the second derivative is maximum
    if len(SSE_d2) > 0:
        elbow_point = np.argmax(SSE_d2) + 1  # +1 because of np.diff reducing array length
    else:
        elbow_point = 0
    return elbow_point

def compute_optimal_clusters(train_data, cluster_list, eval_metric='silhouette', fit_data=None, sample_size=None, normalize_data=False):
    if fit_data is None:
        fit_data = deepcopy(train_data)
    scores = []
    label_sets = {}
    np.random.seed(0)  # For reproducibility
    metric = 'euclidean'
    if normalize_data:
        data = normalize(data)
        fit_data = deepcopy(train_data)
        metric = 'cosine'
    for n_clusters in cluster_list:
        # Cluster with all data
        kmeans, _ = faiss_k_means(train_data, n_clusters, verbose=False)
        D, I = kmeans.index.search(fit_data, 1) # D is the squared distance
        cluster_labels = I[:, 0]
        # kmeans, cluster_labels = normal_k_means(data, n_clusters)
        label_sets[n_clusters] = cluster_labels
        # Evaluate based on the chosen metric
        if eval_metric == 'silhouette':
            score = silhouette_score(fit_data, cluster_labels, metric=metric, sample_size=sample_size)
        elif eval_metric == 'elbow':
            # score = kmeans.inertia_
            score = np.sum(D)
        elif eval_metric == 'davies_bouldin':
            score = davies_bouldin_score(fit_data, cluster_labels)
        else:
            raise ValueError("Invalid evaluation metric")
        scores.append(score)
        # print(f"{eval_metric.capitalize()} Score for {n_clusters} clusters: {score}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_list, scores, marker='o')
    plt.xlabel('Number of Clusters')
    if eval_metric == 'silhouette':
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
    elif eval_metric == 'elbow':
        plt.ylabel('Sum of Squared Distances')
        plt.title('Elbow Method - Sum of Squared Distances vs Number of Clusters')
    elif eval_metric == 'davies_bouldin':
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Davies-Bouldin Score vs Number of Clusters')
    plt.xticks(cluster_list)
    plt.grid(True)
    plt.show()
    if eval_metric in ['silhouette', 'davies_bouldin']:
        best_n_clusters = cluster_list[np.argmax(scores)]
        best_score = max(scores)
    elif eval_metric == 'elbow':
        elbow_index = find_elbow_point(scores)
        best_n_clusters = cluster_list[elbow_index]
        best_score = scores[elbow_index]
    best_labels = label_sets[best_n_clusters]
    
    return best_n_clusters, best_score, best_labels
