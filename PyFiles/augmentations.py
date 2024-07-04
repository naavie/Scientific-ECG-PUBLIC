import numpy as np
import scipy.signal

# Time Warping
# Helps with stretching or compressing the time axis of an ECG signal. This simulates variations in heart rate.
def time_warp(ecg, sigma=0.2):
    factor = np.random.normal(loc=1.0, scale=sigma)
    return scipy.signal.resample(ecg, int(len(ecg) * factor))

# Amplitude Scaling
# Scales the amplitude of the ECG signal, simulating variations in signal strength
def time_warp(ecg, sigma=0.2):
    factor = np.random.normal(loc=1.0, scale=sigma)
    return scipy.signal.resample(ecg, int(len(ecg) * factor))

# Random Noise
# Adding Gaussian noise to help the model generalize better
def add_noise(ecg, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=ecg.shape)
    return ecg + noise

# Baseline Wander
# Simulates the baseline wander, which is a slow drift in the ECG signal
def add_baseline_wander(ecg, amplitude=0.1):
    wander = amplitude * np.sin(2 * np.pi * np.linspace(0, 1, len(ecg)))
    return ecg + wander

# Lead dropout
# Randomly sets some leads to zero. Creates incomplete 12-lead ECGs
def lead_dropout(ecg, p=0.1):
    mask = np.random.binomial(1, 1-p, size=ecg.shape[0])
    return ecg * mask[:, np.newaxis]

# Mixup
# Creates new samples by linearly interpolating between two existing samples
def mixup(ecg1, ecg2, label1, label2, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha)
    mixed_ecg = lambda_ * ecg1 + (1 - lambda_) * ecg2
    mixed_label = lambda_ * label1 + (1 - lambda_) * label2
    return mixed_ecg, mixed_label

# Augmented Dataset
class AugmentedECGDataset(Dataset):
    def __init__(self, ecgs, labels, transform_prob=0.5):
        self.ecgs = ecgs
        self.labels = labels
        self.transform_prob = transform_prob

    def __len__(self):
        return len(self.ecgs)

    def __getitem__(self, idx):
        ecg = self.ecgs[idx]
        label = self.labels[idx]

        if np.random.random() < self.transform_prob:
            ecg = self.transform(ecg)

        return ecg, label

    def transform(self, ecg):
        # Apply random augmentations
        if np.random.random() < 0.5:
            ecg = time_warp(ecg)
        if np.random.random() < 0.5:
            ecg = amplitude_scale(ecg)
        if np.random.random() < 0.5:
            ecg = add_noise(ecg)
        if np.random.random() < 0.5:
            ecg = add_baseline_wander(ecg)
        if np.random.random() < 0.5:
            ecg = lead_dropout(ecg)
        return ecg

# General guidelines for data augmentation:
# Start with a small probability of applying each augmentation and gradually increase it.
# Monitor the validation performance to ensure the augmentations are helping and not hurting.
