# EEG Motor Imagery Classification

## Overview

This project uses machine learning to classify EEG brain signals recorded 
while subjects imagine moving their left or right hand. If a computer can 
detect which hand someone is imagining moving based on their brain waves, 
it opens the door to more advanced prosthetics and BCIs for people who are 
paralyzed or missing limbs. While detecting left vs right hand imagery is 
a small step, it's a direct building block toward that goal.

## Neuroscience Background

EEG (electroencephalogram) is a noninvasive way to record the brain's 
electrical activity using small sensors (electrodes) placed on the scalp. 
It measures the summed electrical fields produced by thousands of neurons 
firing together.

Motor imagery is imagining a physical movement without actually doing it. 
When you imagine moving your hand, your brain produces a measurable response 
called ERD (event-related desynchronization) — alpha and beta wave power 
(8-30Hz) decreases over the motor cortex as the brain prepares for movement.

What makes left vs right classification possible is contralateral organization 
— the brain is cross-wired. Your left motor cortex controls your right hand 
and vice versa. So imagining moving the right hand produces ERD on the left 
side of the scalp (electrode C3), and imagining moving the left hand produces 
ERD on the right side (electrode C4). The classifier learns this spatial 
asymmetry.

## Dataset

PhysioNet EEG Motor Movement/Imagery Dataset. 109 subjects performing a 
motor imagery task — imagining opening and closing their left or right fist 
in 4 second intervals. 3 runs per subject (runs 4, 8, 12), approximately 
15 epochs per run, totaling 4,888 labeled epochs after preprocessing.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/

## Pipeline

Raw .edf files were downloaded for all 109 subjects. Some subjects were 
recorded at 160Hz and others at 128Hz, so all recordings were downsampled 
to 128Hz for consistency.

Each recording was bandpass filtered to 8-30Hz to isolate the alpha and 
beta frequencies relevant to motor imagery, removing low frequency drift 
and high frequency muscle noise.

The continuous signal was then cut into labeled 4 second epochs based on 
the event annotations in each file. T1 events (left fist) were labeled 0 
and T2 events (right fist) were labeled 1. Rest periods (T0) were discarded 
since the classifier only needs to distinguish left from right.

Data was split by subject — subjects 1-87 for training, subjects 88-109 
for testing. This subject-independent split tests whether the model 
generalizes to people it has never seen, which is the real challenge in 
BCI applications.

Training data was z-score normalized (mean 0, std 1) using training 
statistics only. The same normalization was applied to test data to prevent 
data leakage.

Final dataset: 3,889 training epochs, 999 test epochs, each shape (64, 513) 
— 64 electrodes × 513 timepoints.

## Models

### Baseline CNN

A simple convolutional neural network built in PyTorch. Two Conv2D layers 
with deliberately separated kernels — the first (1×64) slides across time 
to learn temporal patterns, the second (64×1) slides across channels to 
learn spatial patterns. This separation mirrors the neuroscience: first 
learn when the ERD happens, then learn where on the scalp it happens.

BatchNorm for training stability, AvgPool to reduce the time dimension, 
Dropout to prevent overfitting, and a single Linear layer outputting a 
left/right probability.

### EEGNet

EEGNet (Lawhern et al. 2018) is an architecture designed specifically for 
EEG. It uses depthwise separable convolutions to learn temporal and spatial 
features independently with far fewer parameters (~2,500 vs ~4 million for 
the baseline CNN). Fewer parameters means less overfitting on the relatively 
small EEG dataset.

## Results

| Model | Epochs | Test Accuracy |
|-------|--------|---------------|
| Baseline CNN (no normalization) | 30 | ~50% |
| Baseline CNN (with normalization) | 30 | ~62-64% |
| EEGNet | 30 | ~69% |
| EEGNet | 100 | ~68.8% (overfit) |
| EEGNet + early stopping | 50 | ~74% |

Chance level is 50%. The jump from 50% to 62% came entirely from adding 
z-score normalization — a good example of preprocessing mattering as much 
as architecture. EEGNet outperformed the baseline CNN despite having far 
fewer parameters, because its architecture is better suited to EEG's 
channels × time structure.

All results use subject-independent evaluation — test subjects were never 
seen during training.

## How to Run

**Install dependencies:**
```bash
pip install mne numpy matplotlib torch
```

**Download and preprocess data:**
```bash
cd src
python preprocessing.py
```

**Train and evaluate:**
```bash
python train.py
```

To switch between models, change the model line in train.py:
```python
model = EEGNet()        # EEGNet (recommended)
model = EEGClassifier() # Baseline CNN
```

## Future Work

- Per-subject normalization instead of global normalization
- Learning rate scheduling to push accuracy further
- Interpretability analysis — which electrodes and time windows does the 
  model actually focus on? Does it match the C3/C4 neuroscience prediction?
- Cross-subject generalization study — how does accuracy scale as you train 
  on more subjects?
- Extending the pipeline to emotion classification from EEG
- Real-time classification on live EEG data