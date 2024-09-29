#### code ref: https://github.com/1Konny/gradcam_plus_plus-pytorch

import torch
import torch.nn.functional as F
from utils.gradcam_utils import *


class GradCAM(object):
    """Calculate GradCAM salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']
        self.device = model_dict['device']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            target_layer = find_mobilenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print(
                    "please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                self.model_arch(torch.zeros(1, 3, *(input_size), device=self.device))

    def forward(self, input, class_idx=None, retain_graph=False, out_size=None):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        if out_size is not None:
            h, w = out_size
        logit = self.model_arch(input)  ### forward (get score)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)   ### backward (get gradients)
        gradients = self.gradients['value']
        activations = self.activations['value']

        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1)
        alpha = alpha.mean(2)

        weights = alpha.view(b, k, 1, 1)

        weighted_feature_map = weights * activations ### save this
        saliency_map_small_1 = weighted_feature_map.sum(1, keepdim=True)
        saliency_map_small = F.relu(saliency_map_small_1)
        saliency_map = F.upsample(saliency_map_small, size=(h, w), mode='bilinear', align_corners=False)

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        return saliency_map, logit, weighted_feature_map, saliency_map_small, gradients, activations

    def __call__(self, input, class_idx=None, retain_graph=False, out_size=None):
        return self.forward(input, class_idx, retain_graph, out_size)