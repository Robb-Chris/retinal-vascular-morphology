"""
Blood Vessel Extraction Pipeline
--------------------------------
Modernized processing module for retinal blood vessel segmentation & quantitative analysis
using classical morphological techniques (Local Adaptive Thresholding and Frangi Vesselness Filter).

Compatible with the Fundus-AVSeg dataset (Nature Scientific Data, 2025).
"""

import os
import sys
import argparse
import numpy as np
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage import io, color
import skimage.filters as filters
import skimage.morphology as morphology
from skimage.measure import label, regionprops

import sknw

# Path configurations
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotation")
RESULTS_DIR = "results"
TRAIN_LIST = os.path.join(DATA_DIR, "training.txt")
TEST_LIST = os.path.join(DATA_DIR, "testing.txt")

DISEASE_MAP = {
    'N': 'Normal',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'A': 'Age-Related Macular Degeneration'
}


def load_split(split="test"):
    """
    Load list of filenames for a given dataset split ('train', 'test', or 'all').
    """
    filenames = []
    if split in ("train", "all"):
        if os.path.exists(TRAIN_LIST):
            with open(TRAIN_LIST, "r") as f:
                filenames.extend([line.strip() for line in f if line.strip()])
    if split in ("test", "all"):
        if os.path.exists(TEST_LIST):
            with open(TEST_LIST, "r") as f:
                filenames.extend([line.strip() for line in f if line.strip()])
    return sorted(list(set(filenames)))


def get_disease_label(filename):
    """
    Extract disease category from filename (e.g., '001_G.png' -> 'Glaucoma').
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    if len(parts) >= 2 and parts[1] in DISEASE_MAP:
        return DISEASE_MAP[parts[1]]
    return "Unknown"


def load_image(filename):
    """
    Load fundus image from data/images directory.
    """
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return io.imread(path)


def load_annotation(filename):
    """
    Load annotation mask from data/annotation directory if available.
    """
    path = os.path.join(ANNOTATIONS_DIR, filename)
    if not os.path.exists(path):
        return None
    return io.imread(path)


def annotation_to_binary_mask(annotation_img):
    """
    Convert Fundus-AVSeg multi-color annotation to a binary vessel mask.
    Non-black pixels (arteries, veins, crossings, uncertain vessels) -> True (1).
    Background -> False (0).
    """
    if annotation_img is None:
        return None
    if annotation_img.ndim == 3:
        # RGB image: non-black pixels are vessels
        return np.any(annotation_img > 15, axis=2)
    else:
        # Grayscale image
        return annotation_img > 15


def extract_green_channel(image):
    """
    Extract green channel from RGB fundus image.
    The green channel exhibits the highest contrast between vessels and background.
    """
    if image.ndim == 3:
        return image[:, :, 1]
    return image


def segment_adaptive_threshold(green_channel, block_size=125, min_size=1500):
    """
    Method 1: Local Adaptive Gaussian Thresholding + Morphological Opening + Small Object Removal.
    """
    # Local adaptive thresholding
    imbw = green_channel < filters.threshold_local(green_channel, block_size, method='gaussian')
    
    # Morphological Opening with octagon(3,3)
    img_open = morphology.opening(imbw, morphology.octagon(3, 3))
    
    # Small object removal
    m1_mask = morphology.remove_small_objects(img_open, max_size=min_size, connectivity=1)
    return m1_mask


def segment_frangi(green_channel, sigmas=np.arange(1, 4, 1), alpha=0.5, beta=0.5):
    """
    Method 2: Frangi Vesselness Filter + Mean Thresholding + Morphological Opening & Closing.
    """
    frangi_img = filters.frangi(
        green_channel, 
        sigmas=sigmas, 
        alpha=alpha, 
        beta=beta, 
        black_ridges=True
    )
    thresh = np.where(frangi_img > np.mean(frangi_img), True, False)
    img_open = morphology.opening(thresh, morphology.disk(3))
    m2_mask = morphology.closing(img_open, morphology.octagon(4, 4))
    return m2_mask


def evaluate_segmentation(pred_mask, gt_mask):
    """
    Calculate classification metrics: Dice Coefficient, Precision, Recall, Accuracy, F1.
    """
    if gt_mask is None or pred_mask is None:
        return {}

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    total_gt = np.sum(gt)
    total_pred = np.sum(pred)

    dice = (2.0 * tp) / (total_pred + total_gt) if (total_pred + total_gt) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1)
    }


def compute_vessel_metrics(binary_mask):
    """
    Calculate total vessel length and width-based length distribution.
    """
    # Morphological opening & thinning
    img_open = morphology.opening(binary_mask.astype(bool), morphology.disk(3))
    img_thin = morphology.thin(img_open)

    # Distance transform on thinned image for length
    dist_thin = scipy.ndimage.distance_transform_edt(img_thin)
    vessel_length = float(np.sum(dist_thin))

    # Distance transform on full binary mask for thickness evaluation
    dist_full = scipy.ndimage.distance_transform_edt(binary_mask)
    vessel_length_full = float(np.sum(dist_full))

    # Length of vessels wider than thresholds
    wide_15_length = float(np.sum(dist_full[dist_full > 15]))
    wide_40_length = float(np.sum(dist_full[dist_full > 40]))

    return {
        "thinned_length": vessel_length,
        "full_dist_sum": vessel_length_full,
        "length_width_gt_15": wide_15_length,
        "length_width_gt_40": wide_40_length,
        "thin_mask": img_thin
    }


def compute_branch_analysis(thin_mask):
    """
    Extract network graph of vessel skeleton using sknw, computing branch lengths & orientations.
    """
    try:
        graph = sknw.build_sknw(thin_mask)
    except Exception as e:
        graph = None

    label_img = label(thin_mask)
    regions = regionprops(label_img)

    branch_lengths = []
    branch_orientations = []
    for r in regions:
        if r.area >= 2:
            branch_lengths.append(r.axis_major_length)
            branch_orientations.append(r.orientation)

    return {
        "graph": graph,
        "branch_lengths": branch_lengths,
        "branch_orientations": branch_orientations,
        "num_branches": len(branch_lengths)
    }


def process_single_image(filename):
    """
    Run full segmentation & analysis pipeline for a single image.
    """
    img = load_image(filename)
    gt_ann = load_annotation(filename)
    gt_mask = annotation_to_binary_mask(gt_ann)
    disease = get_disease_label(filename)

    green = extract_green_channel(img)

    # Method 1 & Method 2
    m1_pred = segment_adaptive_threshold(green)
    m2_pred = segment_frangi(green)

    # Evaluation against GT
    eval_m1 = evaluate_segmentation(m1_pred, gt_mask)
    eval_m2 = evaluate_segmentation(m2_pred, gt_mask)

    # Metrics for M1
    vessel_metrics = compute_vessel_metrics(m1_pred)
    branch_metrics = compute_branch_analysis(vessel_metrics["thin_mask"])

    return {
        "filename": filename,
        "disease": disease,
        "image_shape": img.shape,
        "m1_pred": m1_pred,
        "m2_pred": m2_pred,
        "gt_mask": gt_mask,
        "eval_m1": eval_m1,
        "eval_m2": eval_m2,
        "vessel_metrics": vessel_metrics,
        "branch_metrics": branch_metrics
    }


def process_dataset(split="test", save_vis=False):
    """
    Batch process dataset split and save summary metrics CSV and optional visualisations.
    """
    filenames = load_split(split)
    if not filenames:
        print(f"No images found for split: '{split}'")
        return pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_vis:
        vis_dir = os.path.join(RESULTS_DIR, "visualisations")
        os.makedirs(vis_dir, exist_ok=True)

    records = []
    print(f"Processing {len(filenames)} images from split '{split}'...")

    for fname in tqdm(filenames):
        try:
            res = process_single_image(fname)
            rec = {
                "filename": fname,
                "disease": res["disease"],
                "m1_dice": res["eval_m1"].get("dice", np.nan),
                "m1_precision": res["eval_m1"].get("precision", np.nan),
                "m1_recall": res["eval_m1"].get("recall", np.nan),
                "m1_f1": res["eval_m1"].get("f1", np.nan),
                "m2_dice": res["eval_m2"].get("dice", np.nan),
                "m2_precision": res["eval_m2"].get("precision", np.nan),
                "m2_recall": res["eval_m2"].get("recall", np.nan),
                "m2_f1": res["eval_m2"].get("f1", np.nan),
                "vessel_length": res["vessel_metrics"]["thinned_length"],
                "num_branches": res["branch_metrics"]["num_branches"]
            }
            records.append(rec)

            if save_vis:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                img = load_image(fname)
                axes[0].imshow(img)
                axes[0].set_title("Original")
                axes[0].axis('off')

                if res["gt_mask"] is not None:
                    axes[1].imshow(res["gt_mask"], cmap="gray")
                    axes[1].set_title("Ground Truth")
                axes[1].axis('off')

                axes[2].imshow(res["m1_pred"], cmap="gray")
                axes[2].set_title(f"Method 1 (Dice: {rec['m1_dice']:.3f})")
                axes[2].axis('off')

                axes[3].imshow(res["m2_pred"], cmap="gray")
                axes[3].set_title(f"Method 2 (Dice: {rec['m2_dice']:.3f})")
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{os.path.splitext(fname)[0]}_comparison.png"))
                plt.close(fig)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, f"metrics_{split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Print Summary Statistics
    if not df.empty:
        print("\n--- Summary Performance ---")
        print(f"Method 1 Mean Dice: {df['m1_dice'].mean():.4f} +/- {df['m1_dice'].std():.4f}")
        print(f"Method 2 Mean Dice: {df['m2_dice'].mean():.4f} +/- {df['m2_dice'].std():.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Vessel Extraction & Segmentation Pipeline")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--save-vis", action="store_true", help="Save visualisations to results/visualisations")
    args = parser.parse_args()

    process_dataset(split=args.split, save_vis=args.save_vis)
