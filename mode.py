from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold, RepeatedStratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random

import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A

from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import sklearn

from config import CFG
import sys

sys.path.append('./pytorch-image-models-master')
import timm


class CassvaModel(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x