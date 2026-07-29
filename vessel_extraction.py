"""
Retinal Vascular Morphology Pipeline
------------------------------------
Modernized processing module for retinal blood vessel segmentation & quantitative analysis
using classical morphological techniques (Local Adaptive Thresholding, Frangi Vesselness Filter, and Ensemble Consensus).

Compatible with the Fundus-AVSeg dataset (Nature Scientific Data, 2025).
Includes FOV Retina Masking, CLAHE Contrast Normalization, Otsu Thresholding,
K-Fold Hyperparameter Tuning, and Structural Outlier Detection.
"""

import os
import sys
import argparse
import numpy as np
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage import io, color, exposure
import skimage.filters as filters
from skimage.filters import threshold_otsu, threshold_local
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
        return np.any(annotation_img > 15, axis=2)
    else:
        return annotation_img > 15


def extract_green_channel(image):
    """
    Extract green channel from RGB fundus image.
    The green channel exhibits the highest contrast between vessels and background.
    """
    if image.ndim == 3:
        return image[:, :, 1]
    return image


def create_fov_mask(green_channel, threshold=15, erosion_disk=10):
    """
    Generate Field-Of-View (FOV) retina mask to exclude non-retinal black background.
    """
    fov = green_channel > threshold
    if erosion_disk > 0:
        fov = morphology.erosion(fov, morphology.disk(erosion_disk))
    return fov


def preprocess_image(green_channel, clip_limit=0.02):
    """
    Enhance vessel contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE)
    and compute inverted CLAHE representation.
    """
    green_norm = green_channel.astype(float) / 255.0
    clahe = exposure.equalize_adapthist(green_norm, clip_limit=clip_limit)
    inv_clahe = 1.0 - clahe
    return clahe, inv_clahe


def segment_adaptive_threshold(green_channel, block_size=75, min_size=150, offset=0.015):
    """
    Method 1: CLAHE Contrast Normalization + FOV-Masked Local Adaptive Thresholding.
    """
    fov_mask = create_fov_mask(green_channel)
    clahe, _ = preprocess_image(green_channel)

    if block_size % 2 == 0:
        block_size += 1

    local_thresh = threshold_local(clahe, block_size, method='gaussian')
    m1_raw = (clahe < (local_thresh - offset)) & fov_mask

    m1_clean = morphology.remove_small_objects(m1_raw, max_size=min_size, connectivity=1)
    m1_clean = morphology.closing(m1_clean, morphology.disk(2))
    return m1_clean


def segment_frangi(green_channel, sigmas=np.arange(1, 6, 1), alpha=0.5, beta=0.5, otsu_factor=0.5):
    """
    Method 2: CLAHE + Multi-Scale Frangi Filter + FOV Otsu Thresholding.
    """
    fov_mask = create_fov_mask(green_channel)
    _, inv_clahe = preprocess_image(green_channel)

    frangi_resp = filters.frangi(
        inv_clahe,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        black_ridges=False
    )

    frangi_fov = frangi_resp[fov_mask]
    if len(frangi_fov) > 0:
        otsu_t = threshold_otsu(frangi_fov)
        m2_raw = (frangi_resp > (otsu_t * otsu_factor)) & fov_mask
    else:
        m2_raw = np.zeros_like(green_channel, dtype=bool)

    m2_clean = morphology.remove_small_objects(m2_raw, max_size=150, connectivity=1)
    m2_clean = morphology.closing(m2_clean, morphology.disk(2))
    return m2_clean


def segment_ensemble(m1_pred, m2_pred):
    """
    Method 3: Ensemble Consensus Segmentation (Intersection of Method 1 High-Recall
    and Method 2 High-Precision).
    """
    m3_pred = m1_pred & m2_pred
    m3_clean = morphology.remove_small_objects(m3_pred, max_size=100, connectivity=1)
    return m3_clean


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
    img_open = morphology.opening(binary_mask.astype(bool), morphology.disk(3))
    img_thin = morphology.thin(img_open)

    dist_thin = scipy.ndimage.distance_transform_edt(img_thin)
    vessel_length = float(np.sum(dist_thin))

    dist_full = scipy.ndimage.distance_transform_edt(binary_mask)
    vessel_length_full = float(np.sum(dist_full))

    wide_15_length = float(np.sum(dist_full[dist_full > 15]))
    wide_40_length = float(np.sum(dist_full[dist_full > 40]))

    return {
        "thinned_length": vessel_length,
        "full_dist_sum": vessel_length_full,
        "length_width_gt_15": wide_15_length,
        "length_width_gt_40": wide_40_length,
        "thin_mask": img_thin
    }


def compute_branch_analysis(thin_mask, img_shape=None):
    """
    Extract network graph of vessel skeleton using sknw, computing branch lengths,
    orientations, branch density, and orientation entropy.
    """
    try:
        graph = sknw.build_sknw(thin_mask)
    except Exception:
        graph = None

    label_img = label(thin_mask)
    regions = regionprops(label_img)

    branch_lengths = []
    branch_orientations = []
    for r in regions:
        if r.area >= 2:
            branch_lengths.append(float(r.axis_major_length))
            branch_orientations.append(float(r.orientation))

    num_branches = len(branch_lengths)

    mean_len = float(np.mean(branch_lengths)) if num_branches > 0 else 0.0
    std_len = float(np.std(branch_lengths)) if num_branches > 0 else 0.0
    median_len = float(np.median(branch_lengths)) if num_branches > 0 else 0.0

    if img_shape is not None:
        total_pixels = img_shape[0] * img_shape[1]
        density = (num_branches / total_pixels) * 1e6
    else:
        density = float(num_branches)

    if num_branches > 0:
        counts, _ = np.histogram(branch_orientations, bins=8, range=(-np.pi/2, np.pi/2))
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        orientation_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        orientation_entropy = 0.0

    return {
        "graph": graph,
        "branch_lengths": branch_lengths,
        "branch_orientations": branch_orientations,
        "num_branches": num_branches,
        "mean_branch_length": mean_len,
        "std_branch_length": std_len,
        "median_branch_length": median_len,
        "branch_density": density,
        "orientation_entropy": orientation_entropy
    }


def detect_branch_outliers(branch_lengths, branch_orientations, length_iqr_factor=1.5):
    """
    Identify per-branch outliers based on length IQR within an image.
    """
    if len(branch_lengths) == 0:
        return np.array([]), np.array([])

    lengths = np.array(branch_lengths)
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1
    upper_bound = q3 + (length_iqr_factor * iqr)

    length_outliers = lengths > upper_bound

    orientations = np.array(branch_orientations)
    counts, bin_edges = np.histogram(orientations, bins=8, range=(-np.pi/2, np.pi/2))
    mean_c = np.mean(counts)
    std_c = np.std(counts)
    dense_sectors = counts > (mean_c + 1.5 * std_c)

    return length_outliers, dense_sectors


def detect_image_outliers(df_metrics, iqr_factor=1.5):
    """
    Identify dataset-level image outliers based on structural features.
    """
    df = df_metrics.copy()
    feature_cols = ['vessel_length', 'num_branches', 'branch_density', 'orientation_entropy']

    outlier_flags = pd.Series(False, index=df.index)

    for col in feature_cols:
        if col in df.columns and not df[col].isnull().all():
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - (iqr_factor * iqr)
            upper = q3 + (iqr_factor * iqr)

            col_outliers = (df[col] < lower) | (df[col] > upper)
            df[f'is_outlier_{col}'] = col_outliers
            outlier_flags = outlier_flags | col_outliers

    df['is_overall_outlier'] = outlier_flags
    return df


def plot_outlier_polar_rose(branch_orientations, branch_lengths=None, disease_label="Unknown", ax=None, ref_distribution=None):
    """
    Plot a disease-aware Polar Rose Plot showing orientation distribution and highlighting outliers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

    if len(branch_orientations) == 0:
        ax.set_title(f"Polar Rose ({disease_label}) - No Branches", pad=15)
        return ax

    num_bins = 8
    counts, bin_edges = np.histogram(branch_orientations, bins=num_bins, range=(-np.pi/2, np.pi/2))
    angles = np.linspace(0, 2 * np.pi, num_bins + 1)[:-1]

    mean_c = np.mean(counts)
    std_c = np.std(counts)
    is_sector_outlier = counts > (mean_c + 1.2 * std_c)

    bars = ax.bar(angles, counts, width=0.4, align='center', alpha=0.85)

    for i, (bar, outlier) in enumerate(zip(bars, is_sector_outlier)):
        if outlier:
            bar.set_facecolor('#e74c3c')
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)
        else:
            bar.set_facecolor(plt.cm.viridis(i / num_bins))

    if ref_distribution is not None and len(ref_distribution) == num_bins:
        ax.plot(angles, ref_distribution, color='crimson', linestyle='--', linewidth=2, label='Normal Population Avg')

    ax.set_title(f"Vessel Orientation Polar Rose\n[{disease_label}] (Red = Outlier Clusters)", fontsize=11, fontweight='bold', pad=15)
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(labels)
    ax.set_theta_offset(np.pi / 2)

    return ax


def kfold_splits(filenames, k=5, seed=42):
    """
    Generate k stratified splits based on disease categories.
    """
    np.random.seed(seed)
    disease_groups = {}
    for fn in filenames:
        d = get_disease_label(fn)
        disease_groups.setdefault(d, []).append(fn)

    folds = [[] for _ in range(k)]
    for d, fns in disease_groups.items():
        shuffled = fns.copy()
        np.random.shuffle(shuffled)
        for i, fn in enumerate(shuffled):
            folds[i % k].append(fn)

    splits = []
    for i in range(k):
        val_set = set(folds[i])
        train_set = [fn for fn in filenames if fn not in val_set]
        splits.append((train_set, sorted(list(val_set))))
    return splits


def process_single_image(filename, m1_params=None, m2_params=None):
    """
    Run full segmentation & analysis pipeline for a single image with optional custom parameters.
    """
    img = load_image(filename)
    gt_ann = load_annotation(filename)
    gt_mask = annotation_to_binary_mask(gt_ann)
    disease = get_disease_label(filename)

    green = extract_green_channel(img)

    # Method 1 (CLAHE + FOV Adaptive Thresholding)
    bs = int(m1_params.get("block_size", 75)) if m1_params else 75
    ms = int(m1_params.get("min_size", 150)) if m1_params else 150
    m1_pred = segment_adaptive_threshold(green, block_size=bs, min_size=ms)

    # Method 2 (CLAHE + FOV Frangi Otsu Thresholding)
    a = float(m2_params.get("alpha", 0.5)) if m2_params else 0.5
    b = float(m2_params.get("beta", 0.5)) if m2_params else 0.5
    m2_pred = segment_frangi(green, sigmas=np.arange(1, 6, 1), alpha=a, beta=b)

    # Method 3 (Ensemble Consensus)
    m3_pred = segment_ensemble(m1_pred, m2_pred)

    # Evaluation against GT
    eval_m1 = evaluate_segmentation(m1_pred, gt_mask)
    eval_m2 = evaluate_segmentation(m2_pred, gt_mask)
    eval_m3 = evaluate_segmentation(m3_pred, gt_mask)

    # Vessel Metrics & Structural Branch Analysis (on Ensemble or M1)
    vessel_metrics = compute_vessel_metrics(m3_pred if np.sum(m3_pred) > 50 else m1_pred)
    branch_metrics = compute_branch_analysis(vessel_metrics["thin_mask"], img_shape=img.shape)

    return {
        "filename": filename,
        "disease": disease,
        "image_shape": img.shape,
        "m1_pred": m1_pred,
        "m2_pred": m2_pred,
        "m3_pred": m3_pred,
        "gt_mask": gt_mask,
        "eval_m1": eval_m1,
        "eval_m2": eval_m2,
        "eval_m3": eval_m3,
        "vessel_metrics": vessel_metrics,
        "branch_metrics": branch_metrics
    }


def process_dataset(split="test", save_vis=False, m1_params=None, m2_params=None):
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
            res = process_single_image(fname, m1_params=m1_params, m2_params=m2_params)
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
                "m3_ensemble_dice": res["eval_m3"].get("dice", np.nan),
                "m3_ensemble_precision": res["eval_m3"].get("precision", np.nan),
                "m3_ensemble_recall": res["eval_m3"].get("recall", np.nan),
                "m3_ensemble_f1": res["eval_m3"].get("f1", np.nan),
                "vessel_length": res["vessel_metrics"]["thinned_length"],
                "num_branches": res["branch_metrics"]["num_branches"],
                "mean_branch_length": res["branch_metrics"]["mean_branch_length"],
                "std_branch_length": res["branch_metrics"]["std_branch_length"],
                "median_branch_length": res["branch_metrics"]["median_branch_length"],
                "branch_density": res["branch_metrics"]["branch_density"],
                "orientation_entropy": res["branch_metrics"]["orientation_entropy"]
            }
            records.append(rec)

            if save_vis:
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
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

                axes[4].imshow(res["m3_pred"], cmap="gray")
                axes[4].set_title(f"Ensemble (Dice: {rec['m3_ensemble_dice']:.3f})")
                axes[4].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{os.path.splitext(fname)[0]}_comparison.png"))
                plt.close(fig)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = detect_image_outliers(df)

    csv_path = os.path.join(RESULTS_DIR, f"metrics_{split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    if not df.empty:
        print("\n--- Upgraded Pipeline Performance Summary ---")
        print(f"Method 1 (Adaptive) Mean Dice: {df['m1_dice'].mean():.4f} +/- {df['m1_dice'].std():.4f} (Recall: {df['m1_recall'].mean():.4f})")
        print(f"Method 2 (Frangi)   Mean Dice: {df['m2_dice'].mean():.4f} +/- {df['m2_dice'].std():.4f} (Precision: {df['m2_precision'].mean():.4f})")
        print(f"Method 3 (Ensemble) Mean Dice: {df['m3_ensemble_dice'].mean():.4f} +/- {df['m3_ensemble_dice'].std():.4f} (Precision: {df['m3_ensemble_precision'].mean():.4f})")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Vessel Extraction & Segmentation Pipeline")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--save-vis", action="store_true", help="Save visualisations to results/visualisations")
    args = parser.parse_args()

    process_dataset(split=args.split, save_vis=args.save_vis)
