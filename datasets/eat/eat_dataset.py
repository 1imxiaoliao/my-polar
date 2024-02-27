import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import albumentations.augmentations.functional as F

sys.path.append('..')
import helpers as h
import polar_transformations

# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class EATDataset(Dataset):
  width = 128
  height = 128

  in_channels = 1
  out_channels = 1

  def __init__(self, directory, polar=True, manual_centers=None, center_augmentation=False):
    self.directory = p.join('datasets/eat', directory)
    self.polar = polar
    self.manual_centers = manual_centers
    self.center_augmentation = center_augmentation

    self.file_names = h.listdir(p.join(self.directory, 'label'))
    self.file_names.sort()
    
  def __len__(self):
    #return 16 # overfit single batch
    return len(self.file_names)

  def __getitem__(self, idx):
    file_name = self.file_names[idx]
    label_file = p.join(self.directory, 'label', file_name)
    input_file = p.join(self.directory, 'input', file_name)

    label = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
    label = label.astype(np.float32)
    label /= 255.0
    
    input = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    input = input.astype(np.float32)
    input /= 255.0
    # zero-centered globally because CT machines are calibrated to have even 
    # intensities across images
    input -= GLOBAL_PIXEL_MEAN
    
    # convert to polar
    if self.polar:
      if self.manual_centers is not None:
        center = self.manual_centers[idx]
      else:
        center = polar_transformations.centroid(label)

      if self.center_augmentation and np.random.uniform() < 0.3:
        center_max_shift = 0.05 * EATDataset.height
        center = np.array(center)
        center = (
          center[0] + np.random.uniform(-center_max_shift, center_max_shift),
          center[1] + np.random.uniform(-center_max_shift, center_max_shift))
      
      input = polar_transformations.to_polar(input, center)
      label = polar_transformations.to_polar(label, center)

    # to PyTorch expected format
    input = np.expand_dims(input, axis=-1)
    input = input.transpose(2, 0, 1)
    label = np.expand_dims(label, axis=-1)
    label = label.transpose(2, 0, 1)

    input_tensor = torch.from_numpy(input)
    label_tensor = torch.from_numpy(label)

    return input_tensor, label_tensor
