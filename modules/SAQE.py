# Ultralytics YOLO, AGPL-3.0 license
"""Quality-estimation components extracted from the MED project.

Source component: ultralytics/nn/extra_modules/head.py.
MLP, LQE, Detect_LQE, and Detect_SAQE retain their original class names
and forward behavior. Detect is inherited from standard Ultralytics.

Dependencies: PyTorch and a YOLO11-capable Ultralytics installation.
Register Detect_SAQE in the Ultralytics model parser and detection-head
handling before using it in a model YAML. This extraction does not
include the complete MED detector or preserve old checkpoint module paths.

The current LQE uses top-k peaks, their mean, and normalized entropy
(24 input channels for k=4). Its original compatibility branch remains:
if a restored reg_conf expects 20 channels, entropy is omitted. The
original regression-distribution reshape is also retained.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ultralytics.nn.modules.head import Detect

__all__ = ['MLP', 'LQE', 'Detect_LQE', 'Detect_SAQE']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(nn.Conv2d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LQE(nn.Module):
    """
    Scattering-aware Location Quality Estimator (LQE/SAQE).
    Estimates a localization-quality offset from distribution statistics of
    the bounding-box regression branch.
    """
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        """
        Args:
            k: Number of top probability peaks used for each box side.
            hidden_dim: Hidden dimension of the quality-estimation MLP.
            num_layers: Number of MLP layers.
            reg_max: Number of discrete bins in the DFL box distribution.
        """
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        # Per box side: top-k peaks, their mean, and normalized entropy.
        self.reg_conf = MLP(4 * (k + 2), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        """
        Args:
            scores: Classification logits with shape [B, num_classes, H, W].
            pred_corners: Box distribution logits with shape [B, 4*reg_max, H, W].

        Returns:
            Classification logits calibrated by the localization-quality offset.
        """
        B, _, H, W = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, self.reg_max, 4, H, W), dim=1)
        prob_topk, _ = prob.topk(self.k, dim=1)

        entropy = -(prob * (prob.clamp_min(1e-9)).log()).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.reg_max)

        stat = torch.cat([prob_topk, prob_topk.mean(dim=1, keepdim=True), entropy], dim=1)
        expected_channels = self.reg_conf.layers[0].in_channels
        if stat.reshape(B, -1, H, W).shape[1] != expected_channels:
            legacy_stat = torch.cat([prob_topk, prob_topk.mean(dim=1, keepdim=True)], dim=1)
            if legacy_stat.reshape(B, -1, H, W).shape[1] == expected_channels:
                stat = legacy_stat
            else:
                raise RuntimeError(
                    f"LQE descriptor has {stat.reshape(B, -1, H, W).shape[1]} channels, "
                    f"but reg_conf expects {expected_channels}."
                )
        quality_score = self.reg_conf(stat.reshape(B, -1, H, W))
        return scores + quality_score

class Detect_LQE(Detect):
    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__(nc, ch)
        self.lqe = nn.ModuleList(LQE(4, 64, 2, self.reg_max) for x in ch)

        if self.end2end:
            self.one2one_lqe = copy.deepcopy(self.lqe)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            pred_corners = self.cv2[i](x[i])
            pred_scores = self.lqe[i](self.cv3[i](x[i]), pred_corners)
            x[i] = torch.cat((pred_corners, pred_scores), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        # x_detach = [xi.detach() for xi in x]
        one2one = [
            None for i in range(self.nl)
        ]

        for i in range(self.nl):
            pred_corners = self.one2one_cv2[i](x[i])
            pred_scores = self.one2one_lqe[i](self.one2one_cv3[i](x[i]), pred_corners)
            one2one[i] = torch.cat((pred_corners, pred_scores), 1)

        if hasattr(self, 'cv2') and hasattr(self, 'cv3'):            
            for i in range(self.nl):
                pred_corners = self.cv2[i](x[i])
                pred_scores = self.lqe[i](self.cv3[i](x[i]), pred_corners)
                x[i] = torch.cat((pred_corners, pred_scores), 1)

        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})


class Detect_SAQE(Detect_LQE):
    """Scattering-Aware Quality Estimation head used by the MED detector."""
    pass
