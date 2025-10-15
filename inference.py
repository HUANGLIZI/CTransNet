"""Inference script for TLS prediction and survival analysis."""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from glob import glob
import h5py
from scipy.ndimage import zoom

import Config as config
from nets.CTransNet import CTransNet


def center_crop(img, lab, crop_size):
    """Center crop image and label based on label bounding box."""
    mask = (lab != 0)
    brain_voxels = np.where(mask != 0)
    minXidx = int(np.min(brain_voxels[0]))
    maxXidx = int(np.max(brain_voxels[0]))
    minYidx = int(np.min(brain_voxels[1]))
    maxYidx = int(np.max(brain_voxels[1]))

    x_extend = (crop_size[0] - (maxXidx - minXidx)) // 2
    y_extend = (crop_size[1] - (maxYidx - minYidx)) // 2

    new_minXind = max(0, minXidx - x_extend)
    new_maxXind = min(new_minXind+crop_size[0], img.shape[0])

    new_minYind = max(0, minYidx - y_extend)
    new_maxYind = min(new_minYind+crop_size[1], img.shape[1])

    return img[new_minXind:new_maxXind, new_minYind:new_maxYind], lab[new_minXind:new_maxXind, new_minYind:new_maxYind]


class InferenceDataSet(Dataset):
    """
    Dataset class for inference on TCGA data.
    Loads h5 files from specified directory without requiring labels or csv file.
    """
    def __init__(self, data_dir, output_size=(224, 224)):
        """
        Args:
            data_dir: Directory containing .h5 files
            output_size: Output image size (height, width)
        """
        self.data_dir = data_dir
        self.output_size = output_size

        # Get all h5 files from the directory
        self.h5_files = sorted(glob(os.path.join(data_dir, "*.h5")))

        if len(self.h5_files) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        print(f"Found {len(self.h5_files)} h5 files in {data_dir}")

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]

        # Load h5 file
        with h5py.File(h5_path, "r") as h5f:
            # Load data from h5 file
            image = h5f["image"][:]
            label = h5f["label"][:]
            id_value = h5f["id"][()]
            radiomics = h5f["radiomics_feature"][:]
            clinical = h5f["clinical_feature"][:]

        # Preprocess image (same as validation in BaseDataSet)
        cropped_img, cropped_lab = center_crop(image, label, self.output_size)
        x, y = cropped_img.shape

        # Resize to output size
        image = zoom(cropped_img, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(cropped_lab, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Normalize image
        image[image < -125] = -125
        image[image > 275] = 275
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Convert to 3 channels
        image = np.array([image] * 3)

        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        radiomics = torch.from_numpy(radiomics.astype(np.float32))
        clinical = torch.from_numpy(clinical.astype(np.float32))

        sample = {
            "id": id_value,
            "image": image,
            "label": label,
            "radimocis": radiomics,  # Keep the same key name as in original code
            "clinical": clinical
        }

        return sample


def validate_TLS(model, data_loader, num_classes):
    """
    TLS prediction validation function.
    Returns predictions for each sample with class probabilities.
    """
    model.eval()
    results_list = []
    id_list = []

    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch = sampled_batch['image'].cuda()
            radimocis = sampled_batch['radimocis'].cuda()
            clinical = sampled_batch["clinical"].cuda()

            # Get TLS predictions
            preds = model(volume_batch, radimocis, clinical)
            preds = F.softmax(preds[0], dim=-1)

            pred_cls = preds.data.cpu().numpy()

            results_list.extend(pred_cls)
            # Handle both scalar and array id values
            id_value = sampled_batch['id']
            if isinstance(id_value, torch.Tensor):
                id_list.extend(id_value.data.cpu().numpy())
            else:
                id_list.append(id_value)

    results_arr = np.array(results_list)

    return id_list, results_arr


def validate_survival(model, data_loader):
    """
    Survival analysis validation function.
    Returns survival predictions for each sample.
    """
    model.eval()
    survival_preds = []
    id_list = []

    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch = sampled_batch['image'].cuda()
            radimocis = sampled_batch['radimocis'].cuda()
            clinical = sampled_batch["clinical"].cuda()

            # Get survival predictions (regression output)
            preds = model(volume_batch, radimocis, clinical)[2]

            # Handle both scalar and array id values
            id_value = sampled_batch['id']
            if isinstance(id_value, torch.Tensor):
                id_list.extend(id_value.data.cpu().numpy())
            else:
                id_list.append(id_value)
            survival_preds.append(preds.detach().cpu().numpy().squeeze())

    return id_list, survival_preds


def run_inference(args):
    """
    Main inference function that combines TLS prediction and survival analysis.
    """
    # Setup model
    num_classes = 3
    config_vit = config.get_CTranS_config()

    model = CTransNet(
        config_vit,
        num_classes,
        image_feature_length=args.image_feature_length,
        radiomics_feature_length=args.radiomics_feature_length,
        clinical_feature_length=args.clinical_feature_length,
        feature_planes=args.feature_planes
    ).cuda()

    # Load model weights
    print(f"Loading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))

    # Setup data loader using InferenceDataSet
    print(f"Loading data from: {args.data_path}")
    db_test = InferenceDataSet(data_dir=args.data_path, output_size=(224, 224))
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Running inference on {len(db_test)} samples...")

    # Get TLS predictions
    print("Computing TLS predictions...")
    id_list_tls, tls_preds = validate_TLS(model, test_loader, num_classes)

    # Get survival predictions
    print("Computing survival predictions...")
    id_list_survival, survival_preds = validate_survival(model, test_loader)

    # Ensure ID lists match
    assert id_list_tls == id_list_survival, "ID lists from TLS and survival predictions don't match!"

    # Prepare results
    results = []
    for i, image_id in enumerate(id_list_tls):
        pred_class = np.argmax(tls_preds[i])
        pred_prob_1 = tls_preds[i][0]
        pred_prob_2 = tls_preds[i][1]
        pred_prob_3 = tls_preds[i][2]
        pred_survival = survival_preds[i]

        results.append({
            'ImageID': image_id,
            'pred': pred_class,
            'pred_prob_1': pred_prob_1,
            'pred_prob_2': pred_prob_2,
            'pred_prob_3': pred_prob_3,
            'pred_survival': pred_survival
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Create results directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    output_path = os.path.join(args.output_dir, args.output_filename)
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Total samples processed: {len(results)}")
    print("\nSample results (first 5 rows):")
    print(df.head())

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for TLS prediction and survival analysis')

    # Data parameters
    parser.add_argument('--data_path', type=str,
                        default='/data3/zhanli/lzh/CTransNet/dataset/H5_TCGA/data',
                        help='Path to directory containing h5 files')

    # Model parameters
    parser.add_argument('--model_path', type=str,
                        required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_feature_length', type=int,
                        default=1000,
                        help='Image feature length')
    parser.add_argument('--radiomics_feature_length', type=int,
                        default=584,
                        help='Radiomics feature length')
    parser.add_argument('--clinical_feature_length', type=int,
                        default=9,
                        help='Clinical feature length')
    parser.add_argument('--feature_planes', type=int,
                        default=128,
                        help='Feature planes in the model')

    # Output parameters
    parser.add_argument('--output_dir', type=str,
                        default='/data3/zhanli/lzh/CTransNet/results',
                        help='Directory to save results')
    parser.add_argument('--output_filename', type=str,
                        default='inference_results.csv',
                        help='Output CSV filename')

    # Other parameters
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Run inference
    results_df = run_inference(args)
