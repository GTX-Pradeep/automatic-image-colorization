# Automatic Image Colorization

An automatic grayscale image colorization system that predicts plausible color information without requiring user-provided color hints.

The project follows a classical computer vision and machine learning approach using **YUV color space, SLIC superpixels, FFT-based features, Support Vector Regression (SVR), and Markov Random Field (MRF) smoothing with Iterated Conditional Modes (ICM)**.

An additional feature set using brightness statistics and spatial position is also evaluated as an improvement over the baseline pipeline.

---

## 1. Problem Statement

Colorizing a grayscale image automatically is an ill-posed problem because a single grayscale intensity can correspond to many possible real-world colors.

For example, the same grayscale intensity could represent blue water, green vegetation, or a gray object.

The objective of this project is to automatically predict plausible color information from grayscale images without requiring manual user input.

---

## 2. Project Objective

The system takes a grayscale image as input and predicts its chrominance information to reconstruct a color image.

The main objectives are:

- Convert images into a suitable color representation.
- Segment images into meaningful regions using SLIC superpixels.
- Extract local texture information using 2D FFT.
- Predict chrominance using Support Vector Regression.
- Improve spatial consistency using MRF-based ICM smoothing.
- Evaluate the resulting colorized images quantitatively.

---

## 3. Pipeline

```text
Input RGB Image
       |
       v
Convert RGB -> YUV
       |
       v
Extract Y (Luminance)
       |
       v
SLIC Superpixel Segmentation
       |
       v
10 x 10 Local Patches
       |
       v
2D FFT Feature Extraction
       |
       v
+-------------------------+
|                         |
|   SVR for U channel     |
|   SVR for V channel     |
|                         |
+-------------------------+
       |
       v
Predicted U/V
       |
       v
MRF + ICM Refinement
       |
       v
Combine Y + U + V
       |
       v
YUV -> RGB
       |
       v
Colorized Image
```

---

## 4. Methodology

### 4.1 YUV Color Space

The input image is represented in YUV color space.

- **Y** represents luminance or brightness.
- **U** and **V** represent chrominance or color information.

During colorization, only the Y channel is provided to the model. The system predicts U and V and combines them with Y to reconstruct the RGB image.

### 4.2 SLIC Superpixels

SLIC (Simple Linear Iterative Clustering) is used to divide the luminance image into compact regions called superpixels.

Instead of predicting color independently for every pixel, the model predicts color at the superpixel level.

This reduces the complexity of the problem and provides spatially meaningful regions for later MRF smoothing.

### 4.3 FFT Feature Extraction

For each superpixel, a local 10 x 10 luminance patch is extracted around its centroid.

A 2D Fast Fourier Transform (FFT) is applied to the patch.

The magnitude of the FFT is flattened into a 100-dimensional feature vector representing local frequency and texture information.

### 4.4 Support Vector Regression

Two independent SVR models are trained:

- One predicts the **U chrominance channel**.
- One predicts the **V chrominance channel**.

The features are standardized before being passed to the SVR models.

### 4.5 MRF and ICM Refinement

The initial SVR predictions are refined using a Markov Random Field (MRF).

Neighboring superpixels with similar features are connected in the graph.

Iterated Conditional Modes (ICM) is then used to update the chrominance values so that neighboring regions have more spatially consistent colors.

### 4.6 Additional Feature Improvement

The baseline model uses only FFT features.

An additional experiment extends the feature representation with:

- Mean luminance of each superpixel.
- Standard deviation of luminance.
- Normalized spatial position of the superpixel.

These features provide additional brightness, variation, and spatial information.

---

## 5. Dataset

The project uses the **Landscape Pictures** dataset obtained from Kaggle.

Dataset source:

`arnaud58/landscape-pictures`

The downloaded dataset contains **4319 images**.

For the main experiment:

- Training images: **150**
- Testing images: **40**
- Image width: **256 pixels**
- Superpixels per image: approximately **400**

The dataset is not included in the GitHub repository.

---

## 6. Experiments

Two main configurations were evaluated.

### Baseline Configuration

Features:

- 2D FFT features
- SVR for U/V prediction
- MRF + ICM refinement

### Improved Configuration

Features:

- 2D FFT features
- Mean luminance
- Luminance standard deviation
- Normalized spatial position
- SVR for U/V prediction
- MRF + ICM refinement

---

## 7. Results

### Baseline Experiment

| Method | RGB Distance ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| Grayscale | 0.1203 | 22.4285 | 0.9240 |
| Baseline | 0.1205 | 22.4473 | 0.9234 |
| SVR | 0.1141 | 22.6380 | 0.8818 |
| SVR + MRF/ICM | **0.1125** | **22.7809** | 0.8899 |

### Improved Experiment

| Method | RGB Distance ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| Grayscale | 0.1203 | 22.4285 | 0.9240 |
| Baseline | 0.1205 | 22.4473 | 0.9234 |
| SVR | 0.1097 | 22.9052 | 0.8842 |
| SVR + MRF/ICM | **0.1083** | **23.0412** | **0.8920** |

The improved feature representation reduced RGB distance and increased PSNR and SSIM compared with the original SVR + MRF/ICM configuration.

---

## 8. Project Structure

```text
automatic-image-colorization/
│
├── data/
│   └── raw/
│       └── landscape-pictures/
│
├── models/
│
├── outputs/
│   ├── baseline/
│   ├── svr/
│   └── improved/
│
├── scripts/
│   ├── download_data.py
│   ├── run_experiment.py
│   └── demo.py
│
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   ├── model.py
│   ├── smoothing.py
│   ├── colorization.py
│   └── evaluation.py
│
├── .gitignore
└── README.md
```

---

## 9. Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/GTX-Pradeep/automatic-image-colorization.git
cd automatic-image-colorization

python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-image scikit-learn matplotlib tqdm joblib pillow
```

---

## 10. Dataset Setup

The dataset can be downloaded using the Kaggle CLI:

```bash
kaggle datasets download -d arnaud58/landscape-pictures -p data/raw
```

After downloading, extract the dataset into:

```text
data/raw/landscape-pictures/
```

---

## 11. Running the Experiment

Run the baseline experiment:

```bash
python scripts/run_experiment.py \
    --data data/raw/landscape-pictures \
    --n_train 150 \
    --n_test 40
```

Run the improved feature experiment:

```bash
python scripts/run_experiment.py \
    --data data/raw/landscape-pictures \
    --n_train 150 \
    --n_test 40 \
    --extra \
    --tag _extra
```

The experiment generates:

- Trained model files.
- Per-image evaluation metrics.
- Average evaluation metrics.
- Visual comparison images.

---

## 12. Evaluation Metrics

The project uses three metrics.

### RGB Distance

Measures the average Euclidean distance between predicted and ground-truth RGB pixels.

**Lower is better.**

### PSNR

Peak Signal-to-Noise Ratio measures reconstruction quality.

**Higher is better.**

### SSIM

Structural Similarity Index measures structural similarity between the predicted and ground-truth images.

**Higher is better.**

---

## 13. Visual Demo

After training a model, the demo script can be used to compare the grayscale input, superpixel segmentation, SVR output, and SVR + MRF/ICM output.

```bash
python scripts/demo.py path/to/image.jpg models/colorizer.joblib
```

The visualization is saved as:

```text
demo_output.png
```

---

## 14. Limitations

The current approach has several limitations:

- Colorization can produce block-like artifacts due to superpixel-level predictions.
- Some colors may bleed across neighboring regions.
- The model may predict incorrect colors for ambiguous objects.
- The classical SVR approach has limited global image understanding.
- The quality depends strongly on the training dataset and extracted features.

---

## 15. Future Improvements

Possible future improvements include:

- CNN-based feature extraction.
- Deep learning based color prediction.
- Global image context features.
- Pixel-level color refinement.
- Better multimodal color prediction.
- Alternative color spaces such as CIELAB.
- Larger and more diverse training datasets.

---

## 16. Conclusion

This project demonstrates an automatic image colorization pipeline using classical computer vision and machine learning techniques.

The system combines **YUV decomposition, SLIC superpixels, FFT features, SVR-based chrominance prediction, and MRF/ICM refinement**.

The additional brightness and spatial features improved the quantitative performance of the system, with the best configuration achieving an RGB distance of **0.1083**, PSNR of **23.0412 dB**, and SSIM of **0.8920** on the evaluated test set.

The project provides a complete end-to-end pipeline from grayscale input to reconstructed color output while demonstrating how local image features and spatial relationships can be combined for automatic colorization.