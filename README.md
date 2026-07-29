# Blood Vessel Extraction

A Python framework for retinal blood vessel extraction, segmentation, and structural quantitative analysis from colour fundus images using classical morphological image processing techniques.

This project implements and evaluates **Local Adaptive Thresholding** and the **Frangi Vesselness Filter**, as well as morphological operations (Opening, Closing, White Top-Hat Transform), skeletonization, vessel length/thickness estimation, and network graph branch direction analysis.

It is designed for evaluation on the **Fundus-AVSeg** dataset.

---

## Table of Contents

- [Dataset](#dataset)
- [Annotation Scheme](#annotation-scheme)
- [Morphological Techniques](#morphological-techniques)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Evaluation & Metrics](#evaluation--metrics)
- [Citation & References](#citation--references)

---

## Dataset

This project utilizes the **Fundus-AVSeg** dataset:
- **Paper**: Deng et al., *"A Fundus Image Dataset for AI-based Artery-Vein Vessel Segmentation"*, *Scientific Data*, 12, 2025. [DOI: 10.1038/s41597-025-05381-2](https://www.nature.com/articles/s41597-025-05381-2)
- **Dataset Repository**: Figshare - [Fundus-AVSeg Dataset](https://figshare.com/articles/dataset/Fundus-AVSeg/27938034)

### Dataset Properties
- **100 High-Resolution Fundus Images** captured with ZEISS VISUCAM200 & Canon fundus cameras.
- **Resolutions**: 2656 × 1992 and 1280 × 1280 pixels.
- **Disease Coverage**:
  - `N`: Normal (40 images)
  - `D`: Diabetic Retinopathy (20 images)
  - `G`: Glaucoma (20 images)
  - `A`: Age-Related Macular Degeneration (20 images)
- **Data Split**: Pre-defined split of 80 training images (`data/training.txt`) and 20 testing images (`data/testing.txt`).

*Note: Due to storage size constraints (~200MB), image files (`data/images/`) and manual annotations (`data/annotation/`) are excluded via `.gitignore` and must be downloaded separately from Figshare into the `data/` folder.*

---

## Annotation Scheme

Fundus-AVSeg provides pixel-wise manual annotations color-coded as follows:

| Color | RGB Value | Target Category |
| :--- | :--- | :--- |
| **Red** | `(255, 0, 0)` | Arterial Blood Vessels |
| **Blue** | `(0, 0, 255)` | Venous Blood Vessels |
| **Green** | `(0, 255, 0)` | Artery-Vein Crossings |
| **White** | `(255, 255, 255)` | Vessels of Uncertain Category |
| **Black** | `(0, 0, 0)` | Non-Vessel Background |

For general binary vessel segmentation evaluation, any non-black pixel is converted to a binary vessel mask (`vessel = True`).

---

## Morphological Techniques

1. **Green Channel Extraction**: Retinal fundus images are converted to green channel images, which yield optimal vessel-to-background contrast.
2. **Method 1 - Local Adaptive Thresholding**:
   - Gaussian local adaptive thresholding (block size = 125).
   - Morphological opening with octagonal structuring elements (size = 3x3).
   - Small object removal (connectivity = 1, minimum area = 1500 px).
3. **Method 2 - Frangi Vesselness Filter**:
   - Multi-scale Hessian-based Frangi filter (`sigmas=(0.5, 5)`, `alpha=10`, `beta=15`).
   - Mean thresholding.
   - Morphological opening with disk structuring elements (radius = 3).
   - Morphological closing with octagonal structuring elements (size = 4x4).
4. **Quantitative Analysis**:
   - Thinning / Skeletonization (`skimage.morphology.thin`).
   - Distance Transform (`scipy.ndimage.distance_transform_edt`) for vessel length and width calculation.
   - Network Graph Construction (`sknw`) for branch length and polar rose orientation analysis.

---

## Project Structure

```
Blood-Vessel-Extraction/
├── .gitignore
├── README.md
├── requirements.txt
├── vessel_extraction.py          # Modular Python pipeline & CLI
├── sknw.py                       # Vendored skeleton-to-graph library
├── vessel_segmentation.ipynb     # Interactive inspection & evaluation notebook
├── data/
│   ├── training.txt              # Train split file list (80 images)
│   ├── testing.txt               # Test split file list (20 images)
│   ├── images/                   # Fundus image files (gitignored)
│   └── annotation/               # Pixel-wise annotation masks (gitignored)
└── results/                      # Output metrics CSV & visualisations (gitignored)
    ├── metrics_test.csv
    └── visualisations/
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Robb-Chris/Blood-Vessel-Extraction.git
   cd Blood-Vessel-Extraction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Download the images and annotations from [Figshare](https://figshare.com/articles/dataset/Fundus-AVSeg/27938034) and extract into the `data/` folder:
   - `data/images/001_G.png` ... `100_N.png`
   - `data/annotation/001_G.png` ... `100_N.png`

---

## Usage

### Running the Python Pipeline (CLI)

Batch process images, calculate metrics, and save output visualisations:

```bash
# Evaluate on test split (20 images) and save comparison plots
python vessel_extraction.py --split test --save-vis

# Evaluate on full dataset (100 images)
python vessel_extraction.py --split all
```

Output metrics are saved to `results/metrics_<split>.csv` and plots to `results/visualisations/`.

### Running the Jupyter Notebook

Open `vessel_segmentation.ipynb` in Jupyter Notebook or VS Code to interactively:
- Inspect fundus images and colour-coded ground truth annotations.
- Step through Method 1 and Method 2 intermediate image transformations.
- Compare performance metrics across different disease categories (Normal, DR, Glaucoma, AMD).
- Visualize skeleton graphs and branch direction polar plots.

---

## Evaluation & Metrics

The pipeline calculates the following segmentation and structural metrics:

- **Segmentation Performance**:
  - **Dice Similarity Coefficient**: $\frac{2 \times |P \cap G|}{|P| + |G|}$
  - **Precision**: $\frac{|P \cap G|}{|P|}$
  - **Recall / Sensitivity**: $\frac{|P \cap G|}{|G|}$
  - **F1 Score**
- **Structural Vessel Analysis**:
  - **Total Vessel Length**: Euclidean distance transform sum on thinned skeleton.
  - **Wide Vessel Length**: Total length of vessels with width $>15\text{px}$ and $>40\text{px}$.
  - **Branch Graph**: Branch count, branch lengths, and angular orientation angles.

---

## Citation & References

1. Deng, Z., Gao, W., Gong, Z., Gan, R., Chen, L., Zhang, S., & Ma, L. (2025). *A Fundus Image Dataset for AI-based Artery-Vein Vessel Segmentation*. **Nature Scientific Data**, 12(1), 5381. DOI: [10.1038/s41597-025-05381-2](https://doi.org/10.1038/s41597-025-05381-2).
2. Liu, Y. *sknw: Skeleton to NetworkX Graph Converter*. GitHub: [https://github.com/yxdragon/sknw](https://github.com/yxdragon/sknw).
