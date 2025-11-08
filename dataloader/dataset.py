import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import pandas as pd
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image


def get_breakpoints(df, percentiles: list = [10, 20, 30, 40, 50, 60, 70, 80, 90]) -> np.array:
    """
    Gives the times at which death events occur at given percentile
    parameters:
    df - must contain columns 't' (time) and 'e' (death event)
    percentiles - list of percentages at which breakpoints occur (do not include 0 and 100)
    """
    event_times = df.loc[df['st1'] == 1, 'DFS'].values
    breakpoints = np.percentile(event_times, percentiles)
    breakpoints = np.array([0] + breakpoints.tolist() + [df['DFS'].max()])

    return breakpoints

def make_surv_array(time, event):
    '''
    Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    '''

    breaks = np.array([0.0, 8.0, 12.0, 19.0, 33.0, 115.0])
    n_samples = time.shape[0]
    n_intervals = len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap

    surv_array = np.zeros((n_samples, n_intervals*2))
    for i in range(n_samples):
        if event[i] == 1:
            surv_array[i, 0:n_intervals] = 1.0*(time[i] >= breaks[1:])
            if time[i] < breaks[-1]:
                surv_array[i, n_intervals +
                           np.where(time[i] < breaks[1:])[0][0]] = 1
        else:  # event[i] == 0
            surv_array[i, 0:n_intervals] = 1.0*(time[i] >= breaks_midpoint)

    return surv_array


class BaseDataSet(Dataset):
    def __init__(
        self,
        base_dir=None,
        classes="two",
        split="train",
        transform=None,
        surv="DFS"
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.classes = classes
        self.surv = surv
        csv_file = os.path.join(base_dir, "PRFS_TLS_Clinic.csv")
        df_info = pd.read_csv(csv_file, index_col=None)
        self.sample_list = list(df_info[df_info["split"] == self.split]['id'])
        # self.sample_list = list(df_info['id'])
        print(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(os.path.join(
            self._base_dir, "data/{}.h5".format(case)), "r")

        image = h5f["image"][:]
        label = h5f["label"][:]
        id = h5f["id"][()]
        radimocis = h5f["radiomics_feature"][:]
        clinical = h5f["clinical_feature"][:]
        # ihc = h5f["ihc_8"][:]
        os_ = h5f["os"][:]
        if self.surv == "DFS":
            time = np.array([os_[0]])
            event = np.array([os_[1]])
        elif self.surv == "OS":
            time = np.array([os_[2]])
            event = np.array([os_[3]])
        else:
            print("No support")
        surv_info = make_surv_array(time, event)[0]
        if self.classes == "two":
            diagnosis = h5f["diagnosis_TLS"][:][0]
        else:
            diagnosis = h5f["diagnosis_TLS"][:][1]
        # np.array([time, event])
        if self.split == "train":
            sample = {"id": id, "image": image, "label": label, "radimocis": radimocis,
                      "clinical": clinical, "os": np.array([time, event]), "diagnosis": diagnosis}
            sample = self.transform(sample)
        else:
            output_size = (224, 224)
            cropped_img, cropped_lab = center_crop(
                image, label, output_size)
            x, y = cropped_img.shape

            image = zoom(
                cropped_img, (output_size[0] / x, output_size[1] / y), order=0)
            label = zoom(
                cropped_lab, (output_size[0] / x, output_size[1] / y), order=0)
            image[image < -125] = -125
            image[image > 275] = 275
            image = (image - image.mean())/(image.std() + 1e-8)
            image = np.array([image] * 3)
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.uint8))

            sample = {"id": id,"image": image, "label": label, "radimocis": torch.from_numpy(radimocis.astype(np.float32)),
                      "clinical": torch.from_numpy(clinical.astype(np.float32)), "os": torch.from_numpy(np.array([time, event])),
                      "diagnosis": torch.from_numpy(np.array(diagnosis).astype(np.uint8))}
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(
        image.shape[0], image.shape[1]), -2*sigma, 2*sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def get_bbox(gt):
    mask = (gt != 0)
    brain_voxels = np.where(mask != 0)
    minXidx = int(np.min(brain_voxels[0]))
    maxXidx = int(np.max(brain_voxels[0]))
    minYidx = int(np.min(brain_voxels[1]))
    maxYidx = int(np.max(brain_voxels[1]))
    return minXidx, maxXidx, minYidx, maxYidx


def center_crop(img, lab, crop_size):
    minXidx, maxXidx, minYidx, maxYidx = get_bbox(lab)
    x_extend = (crop_size[0] - (maxXidx - minXidx)) // 2
    y_extend = (crop_size[1] - (maxYidx - minYidx)) // 2

    new_minXind = max(0, minXidx - x_extend)
    new_maxXind = min(new_minXind+crop_size[0], img.shape[0])

    new_minYind = max(0, minYidx - y_extend)
    new_maxYind = min(new_minYind+crop_size[1], img.shape[1])

    return img[new_minXind:new_maxXind, new_minYind:new_maxYind], lab[new_minXind:new_maxXind, new_minYind:new_maxYind]


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, id = sample["image"], sample["label"], sample["id"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        random_factor_size = random.randint(7, 13) / 10

        random_cropped_size = [int(self.output_size[0] * random_factor_size),
                               int(self.output_size[1] * random_factor_size)]
        cropped_img, cropped_lab = center_crop(
            image, label, random_cropped_size)
        x, y = cropped_img.shape

        image = zoom(
            cropped_img, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            cropped_lab, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image[image < -125] = -125
        image[image > 275] = 275
        image = (image - image.mean())/(image.std() + 1e-8)
        if random.random() > 0.5:
            image, label = random_noise(image, label)

        image = np.array([image] * 3)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        new_sample = {}
        new_sample["id"] = id
        new_sample["image"] = image
        new_sample["label"] = label
        new_sample["radimocis"] = torch.from_numpy(
            sample["radimocis"].astype(np.float32))
        # new_sample["ihc"] = torch.from_numpy(
        #     sample["ihc"].astype(np.float32))
        new_sample["clinical"] = torch.from_numpy(
            sample["clinical"].astype(np.float32))
        new_sample["os"] = torch.from_numpy(sample["os"].astype(np.float32))
        new_sample["diagnosis"] = torch.from_numpy(
            np.array(sample["diagnosis"]).astype(np.uint8))
        return new_sample
