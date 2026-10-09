# Facial Age Group Classification

**Four computer vision models that look at a face and predict which age group the person is in, compared on accuracy, speed and size.**

Each face is classified into one of four groups: **Child**, **Young**, **Middle-Aged** or **Senior**. Rather than jumping straight to deep learning, the project compares classic hand-crafted feature methods against a convolutional neural network, to see which trade-offs actually matter.

## Results

Tested on 850 held-out face images (trained on 13,300):

| Model | Accuracy | Inference time (ms/image) | Model size |
|---|---|---|---|
| **HOG + SVM** | **77.5%** | 14.0 | 130 MB |
| CNN | 75.9% | 1.6 | 9.5 MB |
| HOG + MLP | 71.9% | 0.01 | 5.4 MB |
| HOG + LBP + SVM | 71.8% | 13.4 | 139 MB |

**What this shows:**

- **HOG + SVM was the most accurate**, but it is also the largest and one of the slowest, because an SVM keeps many of its training examples around to make predictions.
- **The CNN was almost as accurate** at a fraction of the size and about 9× faster, so it's the most practical choice for a real app.
- **HOG + MLP is tiny and very fast**, but gives up about 6 percentage points of accuracy.
- **Adding LBP texture features didn't help.** The extra preprocessing and features made the SVM slightly worse, a useful reminder that more features aren't automatically better.

Per-group results show **Child** faces are the easiest to recognise (around 0.9 F1), while **Middle-Aged** is the hardest (around 0.6 F1), since it sits between two neighbouring groups that look similar.

## How it works

1. **Load and resize** every face image to 64×64 pixels, with labels read from a text file per split.
2. **Extract features** for the classic models:
   - **HOG** (Histogram of Oriented Gradients) captures the shape and edges of the face, such as wrinkles and the outline of facial features.
   - **LBP** (Local Binary Patterns) captures skin texture. Before extracting it, the images are blurred to reduce noise and contrast-normalised with CLAHE to even out lighting.
3. **Train four models:**
   - an **SVM** on HOG features (RBF kernel, balanced class weights);
   - an **MLP** neural network on HOG features;
   - a **CNN** trained directly on the pixels (two convolution and pooling layers, dropout, balanced class weights to handle the uneven number of images per group);
   - an **SVM** on combined HOG + LBP features.
4. **Compare** accuracy, inference time and file size, and plot sample predictions against the true labels for each model.

**Tech:** Python · OpenCV · scikit-image · scikit-learn · TensorFlow/Keras · NumPy · Matplotlib

## Running it

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Put your dataset in a `CW_Dataset/` folder with this layout:
   ```
   CW_Dataset/
   ├── train/
   │   ├── train_labels.txt   # one line per image: <filename> <label 0-3>
   │   └── *.jpg
   └── test/
       ├── test_labels.txt
       └── *.jpg
   ```
3. Open `age_group_classifier.ipynb`, set `DATA_ROOT` in the first cell to the folder containing `CW_Dataset/`, and run the cells in order. Trained models are saved to `Models/`.

The dataset and the trained models aren't included in this repository: the images are of real people, and two of the models are over GitHub's 100 MB file limit.

## Project structure

```
.
├── age_group_classifier.ipynb   # data loading, features, training, comparison, sample predictions
├── requirements.txt
└── README.md
```

## Author

**Yaseen Mneimneih** · [GitHub](https://github.com/Yassman04) · [LinkedIn](https://www.linkedin.com/in/yaseen-mneimneih-8a9305233/)
