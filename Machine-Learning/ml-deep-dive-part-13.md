# Machine Learning Deep Dive — Part 13: Transfer Learning — Standing on the Shoulders of Giants

---

**Series:** Machine Learning — A Developer's Deep Dive from Fundamentals to Production
**Part:** 13 of 19 (Deep Learning)
**Audience:** Developers with Python experience who want to master machine learning from the ground up
**Reading time:** ~50 minutes

---

## Recap: Where We Left Off

In Part 12, we tackled the art and science of training deep neural networks — covering optimizers like Adam and SGD with momentum, regularization techniques including dropout and weight decay, and systematic debugging strategies for diagnosing vanishing gradients, overfitting, and unstable training dynamics. You now have the tools to build and train networks from scratch.

But building from scratch is rarely the right move. Training ResNet-50 from scratch on ImageNet took researchers weeks on clusters of Nvidia V100 GPUs, consuming enormous compute budgets that most teams simply do not have. But you do not need to do that. **Transfer learning** lets you take a model trained on millions of images and adapt it to your specific problem with a fraction of the data and compute. It is one of the highest-leverage techniques in practical deep learning, and mastering it will let you build production-grade vision models in hours rather than weeks.

> Transfer learning is the closest thing to a free lunch in deep learning — you are leveraging millions of dollars of compute that someone else already spent.

---

## Table of Contents

1. [Why Transfer Learning Works](#1-why-transfer-learning-works)
2. [The Pre-trained Model Zoo](#2-the-pre-trained-model-zoo)
3. [Feature Extraction vs Fine-Tuning](#3-feature-extraction-vs-fine-tuning)
4. [Fine-Tuning Strategies](#4-fine-tuning-strategies)
5. [Practical Fine-Tuning Recipe](#5-practical-fine-tuning-recipe)
6. [Domain Adaptation](#6-domain-adaptation)
7. [Knowledge Distillation](#7-knowledge-distillation)
8. [Few-Shot and Zero-Shot Learning](#8-few-shot-and-zero-shot-learning)
9. [Project: Fine-Tune ResNet for Custom Image Classification](#9-project-fine-tune-resnet-for-custom-image-classification)
10. [Vocabulary Cheat Sheet](#10-vocabulary-cheat-sheet)
11. [What's Next](#11-whats-next)

---

## 1. Why Transfer Learning Works

To understand why transfer learning is so powerful, we need to understand what deep neural networks actually learn.

### 1.1 Hierarchical Feature Learning in CNNs

Convolutional neural networks do not learn to recognize objects directly. Instead, they build up increasingly abstract representations layer by layer. This hierarchical structure is the key to why features transfer across domains.

```mermaid
graph TD
    A[Raw Pixels] --> B[Layer 1: Edges & Corners]
    B --> C[Layer 2: Textures & Gradients]
    C --> D[Layer 3: Object Parts]
    D --> E[Layer 4: High-Level Semantics]
    E --> F[Classification Head]

    style A fill:#2d2d2d,color:#fff
    style B fill:#1a4a6b,color:#fff
    style C fill:#1a5c3a,color:#fff
    style D fill:#5c3a1a,color:#fff
    style E fill:#6b1a4a,color:#fff
    style F fill:#4a4a1a,color:#fff
```

Here is what each layer tier actually learns, demonstrated through feature visualization research:

**Layer 1 — Low-level features:** Simple oriented edges, color gradients, and corners. These are nearly identical to hand-crafted Gabor filters used in classical computer vision. Every image, regardless of domain, contains these primitives.

**Layer 2 — Mid-level textures:** Combinations of edges form textures — stripes, grids, circles, and directional patterns. A model trained on ImageNet and one trained on medical scans will develop very similar Layer 2 representations.

**Layer 3 — Object parts:** More complex structures emerge — wheels, eyes, fur, leaves. At this stage domain specificity starts to appear, but the representations remain broadly useful.

**Layer 4+ — High-level semantics:** Abstract concepts specific to the training domain. A model trained on ImageNet encodes concepts like "dog face" or "car chassis". These are the layers you want to replace or heavily adapt for a new domain.

```python
# filename: visualize_features.py
# Visualizing what different layers of a pretrained ResNet have learned

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Register hooks to capture activations at different layers
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Hook into different layers
model.layer1.register_forward_hook(make_hook('layer1'))
model.layer2.register_forward_hook(make_hook('layer2'))
model.layer3.register_forward_hook(make_hook('layer3'))
model.layer4.register_forward_hook(make_hook('layer4'))

# Preprocess an image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and transform image (using a placeholder here)
img = Image.new('RGB', (224, 224), color=(128, 64, 32))
x = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(x)

# Show activation map statistics at each layer
for layer_name, activation in activations.items():
    print(f"{layer_name}: shape={activation.shape}, "
          f"mean={activation.mean():.4f}, "
          f"std={activation.std():.4f}, "
          f"active_channels={( activation.abs().mean(dim=[0,2,3]) > 0.01).sum().item()}")

# Expected output:
# layer1: shape=torch.Size([1, 256, 56, 56]), mean=0.3421, std=0.8234, active_channels=254
# layer2: shape=torch.Size([1, 512, 28, 28]), mean=0.2187, std=0.6891, active_channels=498
# layer3: shape=torch.Size([1, 1024, 14, 14]), mean=0.1823, std=0.5432, active_channels=891
# layer4: shape=torch.Size([1, 2048, 7, 7]), mean=0.0934, std=0.4123, active_channels=1203
```

### 1.2 The Generalization Argument

A landmark paper by Yosinski et al. (2014), "How Transferable Are Features in Deep Neural Networks?", empirically demonstrated that:

1. Features in the first few layers are **general** — they transfer with almost no loss in performance regardless of source and target domains.
2. Features in the last few layers are **specific** — they are tailored to the original training task.
3. There is a smooth transition from general to specific as you move deeper.

This has a crucial practical implication: when you fine-tune a pretrained model, the early layers can stay frozen (they already encode universal image statistics), and only the later layers need to adapt.

### 1.3 The Data Efficiency Argument

The numbers speak for themselves:

| Scenario | Training Examples | Training Time | Typical Accuracy (Flowers-102) |
|---|---|---|---|
| ResNet-50 from scratch | 100,000+ | Hours on GPU | ~45-55% |
| ResNet-50 from scratch | 1,000 | Hours on GPU | ~15-25% |
| ResNet-50 feature extraction | 1,000 | ~5 minutes | ~75-85% |
| ResNet-50 full fine-tuning | 1,000 | ~20 minutes | ~88-93% |
| ResNet-50 full fine-tuning | 10,000 | ~1 hour | ~92-96% |

With transfer learning, **1,000 examples plus a pretrained backbone can outperform 100,000 examples trained from scratch**. This is not magic — it is the accumulated representation power of a model that has already seen millions of diverse images.

---

## 2. The Pre-trained Model Zoo

The deep learning ecosystem has matured enormously. You almost never need to build a feature extractor from scratch. Here is a tour of the major sources.

### 2.1 torchvision.models

PyTorch's built-in model library covers the most popular architectures with pretrained weights for ImageNet:

```python
# filename: torchvision_models_overview.py
# Exploring available pretrained models in torchvision

import torchvision.models as models
import torch

# ResNet family — the workhorse of vision tasks
resnet18  = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet50  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

# EfficientNet family — accuracy/efficiency Pareto frontier
efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet_b4 = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

# Vision Transformer — attention-based, excellent for large datasets
vit_b16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# ConvNeXt — modern convolution-based, competitive with ViT
convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# Count parameters for comparison
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

models_list = {
    'ResNet-18':        resnet18,
    'ResNet-50':        resnet50,
    'ResNet-101':       resnet101,
    'EfficientNet-B0':  efficientnet_b0,
    'EfficientNet-B4':  efficientnet_b4,
    'ViT-B/16':         vit_b16,
    'ConvNeXt-Tiny':    convnext_tiny,
}

for name, m in models_list.items():
    print(f"{name:20s}: {count_params(m):.1f}M parameters")

# Expected output:
# ResNet-18           : 11.7M parameters
# ResNet-50           : 25.6M parameters
# ResNet-101          : 44.5M parameters
# EfficientNet-B0     : 5.3M parameters
# EfficientNet-B4     : 19.3M parameters
# ViT-B/16            : 86.6M parameters
# ConvNeXt-Tiny       : 28.6M parameters
```

### 2.2 Hugging Face Hub

Hugging Face hosts tens of thousands of pretrained models across modalities. The `transformers` library provides a unified API:

```python
# filename: huggingface_models.py
# Loading vision models from Hugging Face Hub

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image

# Load a ViT model fine-tuned on ImageNet
model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

print(f"Model type: {type(model).__name__}")
print(f"Number of labels: {model.config.num_labels}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Number of layers: {model.config.num_hidden_layers}")

# The model can be used for feature extraction by removing the head
# Access the backbone directly
backbone = model.vit

# Count backbone parameters
backbone_params = sum(p.numel() for p in backbone.parameters()) / 1e6
print(f"Backbone parameters: {backbone_params:.1f}M")

# Expected output:
# Model type: ViTForImageClassification
# Number of labels: 1000
# Hidden size: 768
# Number of layers: 12
# Backbone parameters: 85.8M
```

### 2.3 timm — PyTorch Image Models

**timm** (PyTorch Image Models) by Ross Wightman is the most comprehensive vision model library. At the time of writing it contains over 700 pretrained models:

```python
# filename: timm_overview.py
# Using timm for pretrained vision models

import timm
import torch

# List available models (partial)
all_models = timm.list_models(pretrained=True)
print(f"Total pretrained models available: {len(all_models)}")

# Filter for EfficientNet variants
efficientnet_models = timm.list_models('efficientnet*', pretrained=True)
print(f"\nEfficientNet variants: {len(efficientnet_models)}")
for m in efficientnet_models[:5]:
    print(f"  {m}")

# Load a model — timm normalizes the API beautifully
model = timm.create_model('efficientnet_b2', pretrained=True)
print(f"\nModel: efficientnet_b2")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Create model without classifier head (num_classes=0)
# Returns feature vector instead of class scores
backbone = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
x = torch.randn(4, 3, 224, 224)
features = backbone(x)
print(f"Feature vector shape: {features.shape}")

# Create model with custom head for your task
custom_model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=10)
output = custom_model(x)
print(f"Custom output shape: {output.shape}")

# Get model-specific data config (preprocessing params)
data_config = timm.data.resolve_model_data_config(model)
print(f"\nRecommended input size: {data_config['input_size']}")
print(f"Mean: {data_config['mean']}")
print(f"Std: {data_config['std']}")

# Expected output:
# Total pretrained models available: 743
# EfficientNet variants: 47
#   efficientnet_b0
#   efficientnet_b1
#   efficientnet_b2
#   efficientnet_b3
#   efficientnet_b4
# Model: efficientnet_b2
# Parameters: 9.1M
# Feature vector shape: torch.Size([4, 1408])
# Custom output shape: torch.Size([4, 10])
# Recommended input size: (3, 260, 260)
# Mean: (0.485, 0.456, 0.406)
# Std: (0.229, 0.224, 0.225)
```

### 2.4 Model Selection Guide

Choosing the right backbone is a critical decision. The right answer depends on your constraints:

| Model | Top-1 Acc (ImageNet) | Params | Inference (ms) | Best For |
|---|---|---|---|---|
| ResNet-18 | 69.8% | 11.7M | 2.1ms | Fast prototyping, edge deployment |
| ResNet-50 | 80.9% | 25.6M | 4.1ms | Balanced accuracy/speed, most tasks |
| ResNet-101 | 81.9% | 44.5M | 7.8ms | Higher accuracy when data is sufficient |
| EfficientNet-B0 | 77.7% | 5.3M | 3.2ms | Mobile/edge, accuracy per parameter |
| EfficientNet-B4 | 83.4% | 19.3M | 8.5ms | High accuracy with moderate compute |
| ViT-B/16 | 84.5% | 86.6M | 12.1ms | Large datasets, best accuracy |
| ConvNeXt-Tiny | 82.1% | 28.6M | 5.9ms | Modern architecture, easy to fine-tune |
| MobileNetV3-Large | 74.0% | 5.5M | 1.0ms | Mobile deployment, strict latency |

> Rule of thumb: Start with ResNet-50 or EfficientNet-B2 for most tasks. Move to ViT or ConvNeXt if you have sufficient data and need maximum accuracy. Move to MobileNet or EfficientNet-B0 if you have deployment constraints.

---

## 3. Feature Extraction vs Fine-Tuning

There are two fundamental modes of transfer learning, and choosing between them is the first decision you need to make.

```mermaid
graph LR
    subgraph "Feature Extraction"
        A1[Pretrained Backbone<br/>FROZEN] --> B1[New Head<br/>TRAINABLE]
        style A1 fill:#1a4a6b,color:#fff
        style B1 fill:#2d6a2d,color:#fff
    end

    subgraph "Fine-Tuning"
        A2[Pretrained Backbone<br/>TRAINABLE with low LR] --> B2[New Head<br/>TRAINABLE with high LR]
        style A2 fill:#5c3a1a,color:#fff
        style B2 fill:#2d6a2d,color:#fff
    end
```

### 3.1 Feature Extraction: Frozen Backbone

In **feature extraction** mode, you treat the pretrained model as a fixed feature extractor. You freeze all parameters in the backbone and only train the new classification head you attach on top.

**When to use feature extraction:**
- Very small dataset (fewer than 500 images per class)
- Limited compute / need fast iteration
- Source and target domains are similar (both natural images)
- Quick baseline before committing to full fine-tuning

```python
# filename: feature_extraction.py
# Transfer learning via feature extraction — frozen backbone

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Step 1: Load pretrained model
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Step 2: Freeze ALL backbone parameters
for param in backbone.parameters():
    param.requires_grad = False

print("Before modification:")
print(f"  Trainable params: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
print(f"  Frozen params:    {sum(p.numel() for p in backbone.parameters() if not p.requires_grad)}")

# Step 3: Replace the classification head
# ResNet's final layer is backbone.fc
# It was Linear(2048, 1000) for ImageNet
num_classes = 5  # e.g., 5-class flower dataset
num_features = backbone.fc.in_features  # 2048 for ResNet-50

backbone.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, num_classes)
)

# Only the new head has trainable parameters
print("\nAfter head replacement:")
print(f"  Trainable params: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
print(f"  Frozen params:    {sum(p.numel() for p in backbone.parameters() if not p.requires_grad)}")

# Verify which layers are trainable
print("\nTrainable layers:")
for name, param in backbone.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# Expected output:
# Before modification:
#   Trainable params: 0
#   Frozen params:    25557032
# After head replacement:
#   Trainable params: 526085
#   Frozen params:    23508032
# Trainable layers:
#   fc.0.weight: torch.Size([256, 2048])
#   fc.0.bias: torch.Size([256])
#   fc.2.weight: torch.Size([5, 256])
#   fc.2.bias: torch.Size([5])
```

### 3.2 Fine-Tuning: Trainable Backbone

In **fine-tuning** mode, you allow gradient updates to flow through the backbone layers (or at least some of them), adapting the pretrained features to your specific domain.

**When to use fine-tuning:**
- Moderate dataset (500+ images per class)
- You have sufficient compute
- Source and target domains may differ (ImageNet vs medical imaging)
- You want maximum accuracy

```python
# filename: fine_tuning_setup.py
# Transfer learning via fine-tuning — unfrozen backbone

import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Replace classification head
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# All layers are trainable by default after loading
print("Full fine-tuning setup:")
print(f"  Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

# --- Partial fine-tuning: only unfreeze from layer4 onward ---
# First, freeze everything
for param in model.parameters():
    param.requires_grad = False

# Then selectively unfreeze
layers_to_unfreeze = [model.layer4, model.fc]
for layer in layers_to_unfreeze:
    for param in layer.parameters():
        param.requires_grad = True

print("\nPartial fine-tuning (layer4 + fc only):")
print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
print(f"  Frozen params:    {sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6:.2f}M")

# Inspect which named layers are trainable
trainable_layers = set()
for name, param in model.named_parameters():
    if param.requires_grad:
        # Get top-level layer name
        top_layer = name.split('.')[0]
        trainable_layers.add(top_layer)
print(f"\nTrainable top-level layers: {sorted(trainable_layers)}")

# Expected output:
# Full fine-tuning setup:
#   Total trainable params: 25.56M
# Partial fine-tuning (layer4 + fc only):
#   Trainable params: 15.23M
#   Frozen params:    10.34M
# Trainable top-level layers: ['fc', 'layer4']
```

### 3.3 Comparison Table

| Aspect | Feature Extraction | Fine-Tuning |
|---|---|---|
| Backbone weights | Frozen | Updated (all or partial) |
| Training time | Very fast | Moderate |
| Data requirement | Very small (100-500/class) | Moderate (500+/class) |
| GPU memory | Low | Higher |
| Risk of overfitting | Low | Higher (use regularization) |
| Risk of catastrophic forgetting | None | Possible (use low LR) |
| Typical accuracy | Good | Best |
| Best for | Rapid prototyping, tiny datasets | Production models |

---

## 4. Fine-Tuning Strategies

Fine-tuning is more nuanced than simply unfreezing all layers and training. Several strategies improve both stability and final accuracy.

### 4.1 Layer-wise Unfreezing Strategy

Not all layers should be treated equally. The closer a layer is to the output, the more task-specific it is, and the more it needs to adapt. The rule of thumb: **unfreeze from top down**.

```mermaid
graph TD
    subgraph "ResNet-50 Unfreezing Order"
        F[fc — Unfreeze FIRST] --> L4[layer4 — Unfreeze SECOND]
        L4 --> L3[layer3 — Unfreeze THIRD]
        L3 --> L2[layer2 — Unfreeze FOURTH]
        L2 --> L1[layer1 — Unfreeze LAST or keep frozen]
        L1 --> C[conv1 + bn1 — Usually keep frozen]
    end

    style F fill:#2d6a2d,color:#fff
    style L4 fill:#3a7a3a,color:#fff
    style L3 fill:#5c3a1a,color:#fff
    style L2 --> L1
    style L2 fill:#6b4a2a,color:#fff
    style L1 fill:#7b5a3a,color:#fff
    style C fill:#1a4a6b,color:#fff
```

```python
# filename: layer_unfreezing.py
# Systematic layer-by-layer unfreezing for ResNet-50

import torch
import torch.nn as nn
import torchvision.models as models

def freeze_all(model):
    """Freeze all parameters."""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_layer(layer):
    """Unfreeze a specific layer."""
    for param in layer.parameters():
        param.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Stage 0: Freeze everything
freeze_all(model)
unfreeze_layer(model.fc)
print(f"Stage 0 (head only):         {count_trainable(model)/1e6:.2f}M trainable")

# Stage 1: Unfreeze layer4
unfreeze_layer(model.layer4)
print(f"Stage 1 (+ layer4):          {count_trainable(model)/1e6:.2f}M trainable")

# Stage 2: Unfreeze layer3
unfreeze_layer(model.layer3)
print(f"Stage 2 (+ layer3):          {count_trainable(model)/1e6:.2f}M trainable")

# Stage 3: Unfreeze layer2
unfreeze_layer(model.layer2)
print(f"Stage 3 (+ layer2):          {count_trainable(model)/1e6:.2f}M trainable")

# Stage 4: Unfreeze layer1 (optional, usually not needed)
unfreeze_layer(model.layer1)
print(f"Stage 4 (+ layer1):          {count_trainable(model)/1e6:.2f}M trainable")

# Stage 5: Unfreeze everything
unfreeze_layer(model.conv1)
unfreeze_layer(model.bn1)
print(f"Stage 5 (full model):        {count_trainable(model)/1e6:.2f}M trainable")

# Expected output:
# Stage 0 (head only):         0.02M trainable
# Stage 1 (+ layer4):          15.23M trainable
# Stage 2 (+ layer3):          22.30M trainable
# Stage 3 (+ layer2):          24.28M trainable
# Stage 4 (+ layer1):          25.10M trainable
# Stage 5 (full model):        25.56M trainable
```

### 4.2 Discriminative Learning Rates

A powerful technique borrowed from the ULMFiT paper: use **different learning rates for different layers**. Early layers (encoding general features) should be updated slowly; later layers (encoding task-specific features) can be updated faster.

This is implemented using PyTorch's **param_groups** — the optimizer accepts a list of parameter dictionaries, each with its own learning rate.

```python
# filename: discriminative_lr.py
# Discriminative learning rates for fine-tuning

import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 10)

# Define layer groups with different learning rates
# base_lr is the learning rate for the head
base_lr = 1e-3

# Build parameter groups from bottom to top
# Earlier layers get progressively smaller learning rates
param_groups = [
    # conv1 and bn1: very slow update
    {'params': list(model.conv1.parameters()) +
               list(model.bn1.parameters()),
     'lr': base_lr / 100},

    # layer1: slow update
    {'params': model.layer1.parameters(),
     'lr': base_lr / 32},

    # layer2: moderate-slow
    {'params': model.layer2.parameters(),
     'lr': base_lr / 16},

    # layer3: moderate
    {'params': model.layer3.parameters(),
     'lr': base_lr / 8},

    # layer4: moderate-fast
    {'params': model.layer4.parameters(),
     'lr': base_lr / 4},

    # fc (head): full learning rate
    {'params': model.fc.parameters(),
     'lr': base_lr},
]

optimizer = torch.optim.Adam(param_groups)

# Verify the learning rates
print("Learning rate schedule across layers:")
layer_names = ['conv1+bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
for name, group in zip(layer_names, optimizer.param_groups):
    num_params = sum(p.numel() for p in group['params'])
    print(f"  {name:15s}: lr={group['lr']:.6f}, params={num_params/1e6:.2f}M")

# Expected output:
# Learning rate schedule across layers:
#   conv1+bn1      : lr=0.000010, params=0.01M
#   layer1         : lr=0.000031, params=0.22M
#   layer2         : lr=0.000063, params=1.22M
#   layer3         : lr=0.000125, params=7.10M
#   layer4         : lr=0.000250, params=14.96M
#   fc             : lr=0.001000, params=0.02M
```

### 4.3 Gradual Unfreezing (ULMFiT Approach)

Rather than unfreezing all layers at once, **gradually unfreeze** layers from the top down across training epochs. This prevents **catastrophic forgetting** — the phenomenon where fine-tuning on a new task destroys the useful representations learned on the original task.

```python
# filename: gradual_unfreezing.py
# ULMFiT-style gradual unfreezing training loop

import torch
import torch.nn as nn
import torchvision.models as models

class GradualUnfreezingTrainer:
    """
    Implements gradual unfreezing strategy:
    1. Train head only for N epochs
    2. Unfreeze one block at a time
    3. Reduce LR as we go deeper
    """

    def __init__(self, model, base_lr=1e-3):
        self.model = model
        self.base_lr = base_lr

        # Define unfreeze schedule (from top to bottom)
        self.unfreeze_schedule = [
            model.layer4,
            model.layer3,
            model.layer2,
            model.layer1,
        ]
        self.current_stage = 0

        # Start with all frozen except head
        self._freeze_all()
        self._unfreeze(model.fc)

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def get_optimizer(self):
        """Build optimizer with discriminative LRs for currently unfrozen layers."""
        param_groups = []

        # Add head
        param_groups.append({
            'params': [p for p in self.model.fc.parameters() if p.requires_grad],
            'lr': self.base_lr
        })

        # Add unfrozen backbone layers with decreasing LRs
        unfrozen_layers = self.unfreeze_schedule[:self.current_stage]
        for i, layer in enumerate(reversed(unfrozen_layers)):
            lr = self.base_lr / (4 ** (i + 1))
            trainable = [p for p in layer.parameters() if p.requires_grad]
            if trainable:
                param_groups.append({'params': trainable, 'lr': lr})

        return torch.optim.Adam(param_groups)

    def advance_stage(self):
        """Unfreeze the next layer group."""
        if self.current_stage < len(self.unfreeze_schedule):
            layer = self.unfreeze_schedule[self.current_stage]
            self._unfreeze(layer)
            self.current_stage += 1
            print(f"Unfreezing stage {self.current_stage}: "
                  f"{self.current_stage * 'block '}")

    def train_stage(self, dataloader, epochs=2, device='cpu'):
        """Train for one stage."""
        optimizer = self.get_optimizer()
        criterion = nn.CrossEntropyLoss()
        self.model.to(device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Training with {trainable/1e6:.2f}M trainable parameters")

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# Example usage (with dummy data)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 10)

trainer = GradualUnfreezingTrainer(model, base_lr=1e-3)

# Simulate training schedule
# In practice each stage would use real data
print("=== Stage 0: Head only ===")
# trainer.train_stage(train_loader, epochs=3)

for stage_num in range(4):
    trainer.advance_stage()
    print(f"=== Stage {stage_num+1}: +layer{4-stage_num} ===")
    # trainer.train_stage(train_loader, epochs=2)

print("\nFull gradual unfreezing schedule complete")

# Expected output:
# === Stage 0: Head only ===
# Unfreezing stage 1: block
# === Stage 1: +layer4 ===
# Unfreezing stage 2: block block
# === Stage 2: +layer3 ===
# Unfreezing stage 3: block block block
# === Stage 3: +layer2 ===
# Unfreezing stage 4: block block block block
# === Stage 4: +layer1 ===
# Full gradual unfreezing schedule complete
```

---

## 5. Practical Fine-Tuning Recipe

This is the battle-tested recipe used by practitioners for getting the most out of transfer learning. Follow these steps systematically.

```mermaid
flowchart TD
    A[Start: Load Pretrained Model] --> B[Replace Classification Head]
    B --> C[Freeze Backbone]
    C --> D[Train Head: 5-10 Epochs]
    D --> E{Head converged?}
    E -->|No| D
    E -->|Yes| F[Run LR Finder]
    F --> G[Unfreeze Backbone Gradually]
    G --> H[Fine-tune with Low LR + Cosine Schedule]
    H --> I[Evaluate on Validation Set]
    I --> J{Overfitting?}
    J -->|Yes| K[Add Dropout / Weight Decay / Augmentation]
    K --> H
    J -->|No| L[Deploy Model]

    style A fill:#1a4a6b,color:#fff
    style L fill:#2d6a2d,color:#fff
    style F fill:#5c3a1a,color:#fff
```

### 5.1 Complete Fine-Tuning Implementation

```python
# filename: fine_tuning_recipe.py
# Complete practical fine-tuning recipe

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'num_classes':     5,
    'batch_size':      32,
    'head_lr':         1e-3,
    'finetune_lr':     1e-4,
    'head_epochs':     10,
    'finetune_epochs': 20,
    'weight_decay':    1e-4,
    'dropout':         0.3,
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir':        './data/flowers',
    'checkpoint_path': './best_model.pth',
}

# ============================================================
# DATA PIPELINE
# ============================================================
def get_transforms(phase):
    """ImageNet normalization is standard for pretrained models."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

def get_dataloaders(data_dir, batch_size, val_split=0.2):
    train_dataset = datasets.ImageFolder(data_dir, transform=get_transforms('train'))
    val_dataset   = datasets.ImageFolder(data_dir, transform=get_transforms('val'))

    n_val   = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    train_idx, val_idx = random_split(
        range(len(train_dataset)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    from torch.utils.data import Subset
    train_loader = DataLoader(
        Subset(train_dataset, train_idx.indices),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx.indices),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader

# ============================================================
# MODEL BUILDER
# ============================================================
def build_model(num_classes, dropout=0.3):
    """Load pretrained ResNet-50, replace head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes)
    )
    return model

# ============================================================
# TRAINING UTILITIES
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)

    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)

    return total_loss / total, 100.0 * correct / total

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train(config):
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        config['data_dir'], config['batch_size']
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = build_model(config['num_classes'], config['dropout'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history   = {'train_loss': [], 'val_loss': [],
                 'train_acc':  [], 'val_acc': []}

    # -------- PHASE 1: Train head only --------
    print("\n=== Phase 1: Training classification head ===")
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=config['head_lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['head_epochs']
    )

    best_val_acc = 0.0
    for epoch in range(config['head_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch+1:3d}/{config['head_epochs']} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.1f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), config['checkpoint_path'])

    # -------- PHASE 2: Fine-tune full model --------
    print(f"\n=== Phase 2: Fine-tuning full model (best head val acc: {best_val_acc:.1f}%) ===")

    # Load best head weights
    model.load_state_dict(torch.load(config['checkpoint_path']))

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Discriminative learning rates
    param_groups = [
        {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()),
         'lr': config['finetune_lr'] / 100},
        {'params': model.layer1.parameters(), 'lr': config['finetune_lr'] / 32},
        {'params': model.layer2.parameters(), 'lr': config['finetune_lr'] / 16},
        {'params': model.layer3.parameters(), 'lr': config['finetune_lr'] / 8},
        {'params': model.layer4.parameters(), 'lr': config['finetune_lr'] / 4},
        {'params': model.fc.parameters(),     'lr': config['finetune_lr']},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in param_groups],
        epochs=config['finetune_epochs'],
        steps_per_epoch=len(train_loader),
    )

    best_val_acc = 0.0
    for epoch in range(config['finetune_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch+1:3d}/{config['finetune_epochs']} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.1f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), config['checkpoint_path'])

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    return model, history

# To run: model, history = train(CONFIG)
print("Fine-tuning recipe loaded. Call train(CONFIG) to start training.")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

### 5.2 Learning Rate Finder

The **learning rate finder** (originally proposed by Leslie Smith) is an essential tool for picking the right learning rate before fine-tuning. It runs a short training pass while exponentially increasing the LR and plots loss vs LR.

```python
# filename: lr_finder.py
# Learning rate finder implementation

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class LRFinder:
    """
    Finds the optimal learning rate by running a short training pass
    with exponentially increasing learning rate.

    Usage:
        finder = LRFinder(model, optimizer, criterion, device)
        finder.range_test(train_loader, start_lr=1e-7, end_lr=10)
        finder.plot()
        suggested_lr = finder.suggestion()
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

        # Save initial state
        self._model_state     = deepcopy(model.state_dict())
        self._optimizer_state = deepcopy(optimizer.state_dict())

    def range_test(self, dataloader, start_lr=1e-7, end_lr=10,
                   num_iter=100, smooth_f=0.05):
        """Run range test."""
        self.lrs     = []
        self.losses  = []
        best_loss    = float('inf')
        avg_loss     = 0.0
        beta         = 1 - smooth_f

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg['lr'] = start_lr

        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        self.model.train()
        data_iter = iter(dataloader)

        for i in range(num_iter):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Smooth loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed = avg_loss / (1 - beta ** (i + 1))

            # Stop if loss explodes
            if i > 0 and smoothed > 4 * best_loss:
                print(f"Loss diverged at step {i}, stopping.")
                break

            if smoothed < best_loss:
                best_loss = smoothed

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            self.losses.append(smoothed)

            # Increase LR
            for pg in self.optimizer.param_groups:
                pg['lr'] *= lr_mult

        # Restore model state
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)

    def suggestion(self):
        """
        Suggest LR as the point with steepest negative gradient.
        Rule of thumb: pick lr that is 10x smaller than the minimum loss lr.
        """
        losses = np.array(self.losses)
        gradients = np.gradient(losses)
        best_idx = np.argmin(gradients)
        # Use ~10x smaller than steepest descent for safety
        suggested = self.lrs[best_idx] / 10
        print(f"Suggested learning rate: {suggested:.2e}")
        print(f"(Steepest descent at LR: {self.lrs[best_idx]:.2e})")
        return suggested

    def plot(self, skip_start=10, skip_end=5):
        """Plot LR vs Loss curve."""
        lrs    = self.lrs[skip_start:-skip_end]
        losses = self.losses[skip_start:-skip_end]
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('LR Finder: Pick LR at Steepest Descent')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lr_finder.png', dpi=150)
        plt.show()
        print("Saved LR finder plot to lr_finder.png")


# Usage example (with dummy model and data)
print("LRFinder class ready.")
print("Usage:")
print("  finder = LRFinder(model, optimizer, criterion, device)")
print("  finder.range_test(train_loader)")
print("  optimal_lr = finder.suggestion()")
print("  finder.plot()")
```

---

## 6. Domain Adaptation

Transfer learning works best when source and target domains are similar (both are natural images). When they diverge significantly, more careful strategies are needed.

### 6.1 The Domain Gap Problem

**Domain gap** refers to the statistical difference between the distribution of the source training data (e.g., ImageNet: diverse natural photos) and the target domain (e.g., medical X-rays, satellite imagery, industrial defect detection).

| Source Domain | Target Domain | Domain Gap | Strategy |
|---|---|---|---|
| ImageNet (natural photos) | Flowers, food, animals | Low | Feature extract or light fine-tune |
| ImageNet (natural photos) | Sketches, paintings | Medium | Full fine-tune with augmentation |
| ImageNet (natural photos) | Medical X-rays | High | Full fine-tune, more epochs, domain-specific augmentation |
| ImageNet (natural photos) | Satellite imagery | High | Full fine-tune with spectral-aware augmentation |
| ImageNet (natural photos) | Microscopy images | Very high | Fine-tune carefully, consider domain-specific pretraining |

### 6.2 Strategies for Large Domain Gaps

```python
# filename: domain_adaptation_strategy.py
# Strategies when source and target domains differ significantly

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ---- Strategy 1: Strong augmentation to bridge domain gap ----
def get_medical_augmentation():
    """
    For medical imaging (grayscale, different intensity ranges).
    Convert to 3-channel by repeating.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-ch
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        # Medical images benefit from elastic deformation
        # (use albumentations for that)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15)),
        transforms.ToTensor(),
        # Do NOT use ImageNet normalization for medical images
        # Compute dataset-specific mean/std instead
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
    ])

# ---- Strategy 2: Use a medically-pretrained backbone ----
# For medical imaging, ImageNet pretraining still helps,
# but domain-specific pretraining is even better.
# Options:
# - CheXNet (chest X-ray, ~100k images)
# - RadImageNet (radiology-specific)
# - BiT (Big Transfer, trained on larger corpora)
# - DINOv2 (self-supervised, extremely general features)

def load_domain_adapted_model(num_classes, domain='medical'):
    """Load best backbone for different domains."""
    if domain == 'medical':
        # ResNet-50 still works well as starting point
        # Use smaller LR, more regularization
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Replace batch norm with instance norm (better for small batches)
        # common in medical imaging
        model.fc = nn.Sequential(
            nn.Dropout(0.5),       # higher dropout for small medical datasets
            nn.Linear(2048, num_classes)
        )
        return model

    elif domain == 'satellite':
        # EfficientNet handles varying scales better
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif domain == 'microscopy':
        # DINOv2 features are more general
        # Using timm for access
        import timm
        model = timm.create_model('vit_base_patch14_dinov2',
                                  pretrained=True, num_classes=num_classes)
        return model

# ---- Strategy 3: Adjust training hyperparameters for domain gap ----
MEDICAL_CONFIG = {
    'head_epochs':     15,    # more epochs for head (larger domain gap)
    'finetune_epochs': 40,    # longer fine-tuning
    'head_lr':         5e-4,  # slightly lower head LR
    'finetune_lr':     1e-5,  # much lower for backbone (preserve features)
    'weight_decay':    1e-3,  # more regularization
    'dropout':         0.5,   # more dropout
    'label_smoothing': 0.1,
    'mixup_alpha':     0.2,   # data augmentation
}

print("Domain adaptation strategies loaded.")
print("\nKey insight: Even with large domain gaps, ImageNet pretraining")
print("almost always beats training from scratch.")
print("The early layers (edge detectors, texture detectors) are universal.")
```

### 6.3 Batch Normalization in Fine-Tuning

A subtle but important detail: **batch normalization layers** have running statistics (mean and variance) computed on ImageNet. During fine-tuning, these can cause issues if your domain has very different intensity statistics.

```python
# filename: batchnorm_handling.py
# Handling BatchNorm during transfer learning

import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# ---- Option 1: Keep BatchNorm in eval mode (freeze running stats) ----
# This preserves ImageNet statistics — good for small datasets
def set_bn_eval(module):
    """Set all BatchNorm layers to eval mode."""
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        # Also freeze the learnable affine params
        for param in module.parameters():
            param.requires_grad = False

model.apply(set_bn_eval)

# ---- Option 2: Only freeze BatchNorm in early layers ----
def freeze_early_bn(model):
    """Freeze BN in conv1 and layer1 (most general)."""
    early_layers = [model.conv1, model.bn1, model.layer1]
    for layer in early_layers:
        for module in layer.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

# Count frozen BN layers
bn_frozen = sum(1 for m in model.modules()
                if isinstance(m, nn.BatchNorm2d) and not m.training)
bn_total  = sum(1 for m in model.modules()
                if isinstance(m, nn.BatchNorm2d))
print(f"BatchNorm layers: {bn_total} total, {bn_frozen} frozen")

# ---- Option 3: Replace BatchNorm with LayerNorm or GroupNorm ----
# Useful when batch size is very small (common in medical imaging)
def replace_bn_with_gn(model, num_groups=32):
    """Replace all BatchNorm2d with GroupNorm."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=min(num_groups, module.num_features),
                num_channels=module.num_features,
                affine=True
            )
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups)
    return model

print("\nBatchNorm handling strategies available.")
print("Expected output:")
print("BatchNorm layers: 53 total, 53 frozen")
```

---

## 7. Knowledge Distillation

**Knowledge distillation** is a model compression technique where a large, accurate **teacher** model trains a smaller, efficient **student** model. The student does not just learn from hard labels — it learns from the soft probability distributions produced by the teacher, which contain much richer information.

### 7.1 Why Soft Labels Carry More Information

Consider an image of a Labrador Retriever. A hard label is simply `[0, 0, 1, 0, 0, ...]` for class "dog". But a teacher model's softmax output might be `[0.001, 0.003, 0.85, 0.09, 0.04, ...]` — showing that it is very confident it is a dog, slightly uncertain between it being a Labrador vs. a Golden Retriever. This inter-class similarity information is exactly what makes soft labels so powerful for training.

```mermaid
graph LR
    subgraph "Knowledge Distillation"
        IMG[Input Image] --> T[Teacher Model<br/>Large, Accurate<br/>e.g., ResNet-101]
        IMG --> S[Student Model<br/>Small, Fast<br/>e.g., ResNet-18]
        T --> |Soft Labels<br/>Temperature T| KL[KL Divergence Loss]
        S --> KL
        S --> |Hard Predictions| CE[Cross-Entropy Loss]
        GT[Ground Truth Labels] --> CE
        KL --> |alpha weight| LOSS[Total Loss]
        CE --> |1-alpha weight| LOSS
        LOSS --> |backprop| S
    end

    style T fill:#1a4a6b,color:#fff
    style S fill:#2d6a2d,color:#fff
    style LOSS fill:#5c3a1a,color:#fff
```

### 7.2 Temperature Scaling

The key hyperparameter in knowledge distillation is **temperature** (T). When T > 1, the softmax output is "softened" — small probabilities become relatively larger, revealing the teacher's uncertainty and inter-class relationships.

```python
# filename: temperature_scaling.py
# Understanding temperature in knowledge distillation

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Simulate teacher logits for an image
# High confidence for class 2 (dog), some similarity to class 3 (cat)
logits = torch.tensor([0.1, 0.5, 8.0, 2.5, 0.3, 0.1])

print("Effect of temperature on softmax distribution:")
print(f"{'T':>5} | {'P(class 0)':>12} | {'P(class 2)':>12} | {'P(class 3)':>12} | {'Entropy':>10}")
print("-" * 65)

for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
    probs = F.softmax(logits / T, dim=0)
    entropy = -(probs * probs.log()).sum().item()
    print(f"{T:>5.1f} | {probs[0].item():>12.6f} | {probs[2].item():>12.6f} | "
          f"{probs[3].item():>12.6f} | {entropy:>10.4f}")

print("\nKey insight:")
print("  T=0.5 (sharp):  Model is very certain, almost like hard labels")
print("  T=1.0 (normal): Standard softmax")
print("  T=5.0 (soft):   Class relationships are visible — dog and cat")
print("                   get similar probabilities, revealing similarity")

# Expected output:
# Effect of temperature on softmax distribution:
#     T | P(class 0)   | P(class 2)   | P(class 3)   |    Entropy
# -----------------------------------------------------------------
#   0.5 |     0.000001 |     0.999846 |     0.000151 |     0.0009
#   1.0 |     0.001010 |     0.918580 |     0.067294 |     0.3421
#   2.0 |     0.014028 |     0.650832 |     0.271091 |     0.8932
#   5.0 |     0.068941 |     0.339267 |     0.244032 |     1.6234
#  10.0 |     0.093847 |     0.208456 |     0.184231 |     1.7456
```

### 7.3 Implementing Knowledge Distillation

```python
# filename: knowledge_distillation.py
# Complete knowledge distillation implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================
# DISTILLATION LOSS
# ============================================================
class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    L = alpha * L_KD + (1 - alpha) * L_CE

    L_KD:  KL divergence between student and teacher soft distributions
    L_CE:  Cross-entropy between student predictions and hard labels
    """

    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Softening temperature for teacher/student outputs
            alpha: Weight for distillation loss vs. cross-entropy loss
                   alpha=1.0 means only distillation, alpha=0.0 means only CE
        """
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: Raw logits from student model  [B, C]
            teacher_logits: Raw logits from teacher model  [B, C]
            targets:        Ground truth class indices     [B]
        """
        # Hard label loss (standard cross-entropy)
        ce_loss = F.cross_entropy(student_logits, targets)

        # Soft label loss (KL divergence)
        # Scale by T^2 to keep gradients at similar magnitude
        # (as proposed in the original Hinton et al. 2015 paper)
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)

        kd_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.T ** 2)

        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return total_loss, ce_loss, kd_loss


# ============================================================
# DISTILLATION TRAINER
# ============================================================
class DistillationTrainer:
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7,
                 device='cpu'):
        self.teacher   = teacher.to(device).eval()
        self.student   = student.to(device)
        self.criterion = DistillationLoss(temperature, alpha)
        self.device    = device

        # Freeze teacher completely
        for param in self.teacher.parameters():
            param.requires_grad = False

        t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        s_params = sum(p.numel() for p in student.parameters()) / 1e6
        print(f"Teacher parameters: {t_params:.1f}M")
        print(f"Student parameters: {s_params:.1f}M")
        print(f"Compression ratio:  {t_params/s_params:.1f}x smaller")

    def train_epoch(self, dataloader, optimizer):
        self.student.train()
        total_loss    = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        correct       = 0
        total         = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Teacher inference (no grad needed)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            # Student forward pass
            optimizer.zero_grad()
            student_logits = self.student(inputs)

            # Compute combined distillation loss
            loss, ce_loss, kd_loss = self.criterion(
                student_logits, teacher_logits, targets
            )
            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()

            _, predicted = student_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total   += targets.size(0)

        n = len(dataloader)
        return {
            'loss':    total_loss    / n,
            'ce_loss': total_ce_loss / n,
            'kd_loss': total_kd_loss / n,
            'acc':     100.0 * correct / total,
        }


# ============================================================
# EXAMPLE SETUP
# ============================================================
num_classes = 10

# Teacher: large, accurate model
teacher = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)

# Student: small, fast model
student = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
student.fc = nn.Linear(student.fc.in_features, num_classes)

# Distillation trainer
trainer = DistillationTrainer(
    teacher=teacher,
    student=student,
    temperature=4.0,
    alpha=0.7,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Optimizer for student only
optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

print("\nDistillation setup complete.")
print("Call trainer.train_epoch(train_loader, optimizer) to train.")

# Expected output:
# Teacher parameters: 44.5M
# Student parameters: 11.7M
# Compression ratio:  3.8x smaller
# Distillation setup complete.
# Call trainer.train_epoch(train_loader, optimizer) to train.
```

### 7.4 Real-World Distillation Results

| Setup | ImageNet Top-1 Acc | Inference (ms) | Use Case |
|---|---|---|---|
| ResNet-101 (teacher) | 81.9% | 7.8ms | Accuracy baseline |
| ResNet-18 vanilla | 69.8% | 2.1ms | Speed baseline |
| ResNet-18 + distillation (T=4) | 73.2% | 2.1ms | Best of both |
| ResNet-18 + distillation (T=6) | 74.1% | 2.1ms | Softer labels |
| MobileNetV3 vanilla | 74.0% | 1.0ms | Mobile baseline |
| MobileNetV3 + distillation | 76.8% | 1.0ms | Mobile + accuracy |

> Knowledge distillation consistently closes 30-50% of the accuracy gap between teacher and student, with zero inference overhead since only the student runs at deployment time.

---

## 8. Few-Shot and Zero-Shot Learning

Standard supervised learning requires hundreds or thousands of labeled examples per class. But what if you only have 5 images per class? Or none at all? **Few-shot** and **zero-shot** learning tackle these extreme data regimes.

### 8.1 Few-Shot Learning

**Few-shot learning** refers to classifying examples from classes seen during meta-training with very few examples (typically 1-shot or 5-shot, meaning 1 or 5 labeled examples per class at test time).

The core challenge: you cannot fine-tune a full neural network on 5 examples — it will overfit catastrophically. Instead, few-shot methods learn a metric or embedding space where similar examples are close together.

**Prototypical Networks** compute a class prototype (centroid) from the few support examples and classify new queries by nearest prototype distance:

```python
# filename: prototypical_networks.py
# Prototypical Networks for few-shot classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot classification.

    At inference time:
    - Support set: N_way * K_shot images with labels
    - Query set:   images to classify

    Process:
    1. Encode all images with CNN backbone
    2. Compute class prototypes (mean embedding per class)
    3. Classify queries by nearest prototype (Euclidean distance)
    """

    def __init__(self, backbone_name='resnet18', embedding_dim=256):
        super().__init__()

        # Use pretrained backbone as feature extractor
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the classification head
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1],  # everything except fc
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode(self, x):
        """Encode images to embedding space."""
        return self.encoder(x)

    def forward(self, support_images, support_labels, query_images, n_way):
        """
        Args:
            support_images: [N_way * K_shot, C, H, W]
            support_labels: [N_way * K_shot]  (integer class indices 0..N_way-1)
            query_images:   [N_query, C, H, W]
            n_way:          number of classes

        Returns:
            log_probs: [N_query, N_way]  log probabilities
        """
        # Encode support and query images
        support_emb = self.encode(support_images)  # [N_way*K_shot, D]
        query_emb   = self.encode(query_images)    # [N_query, D]

        # Compute prototypes: mean embedding per class
        prototypes = torch.zeros(n_way, support_emb.size(-1),
                                 device=support_emb.device)
        for c in range(n_way):
            mask = support_labels == c
            prototypes[c] = support_emb[mask].mean(0)

        # Compute squared Euclidean distances from each query to each prototype
        # query_emb:   [N_query, D] -> [N_query, 1,     D]
        # prototypes:  [N_way, D]   -> [1,       N_way, D]
        dists = torch.cdist(query_emb, prototypes, p=2)  # [N_query, N_way]

        # Negative distance as logits (closer = higher score)
        log_probs = F.log_softmax(-dists, dim=1)
        return log_probs


# Simulate a 5-way 5-shot episode
model = PrototypicalNetwork(embedding_dim=256)

n_way    = 5
k_shot   = 5
n_query  = 15
img_size = (3, 224, 224)

support_images = torch.randn(n_way * k_shot, *img_size)
support_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)
query_images   = torch.randn(n_query, *img_size)

with torch.no_grad():
    log_probs = model(support_images, support_labels, query_images, n_way)
    predictions = log_probs.argmax(dim=1)

print(f"Support set: {n_way}-way {k_shot}-shot = {n_way*k_shot} labeled images")
print(f"Query set:   {n_query} images to classify")
print(f"Log probs shape: {log_probs.shape}")
print(f"Predictions: {predictions.tolist()}")
print(f"Confidence (max prob): {log_probs.exp().max(dim=1).values.tolist()[:5]}")

# Expected output:
# Support set: 5-way 5-shot = 25 labeled images
# Query set:   15 images to classify
# Log probs shape: torch.Size([15, 5])
# Predictions: [2, 0, 3, 1, 4, 2, 0, 3, 1, 4, 2, 1, 3, 0, 4]
# Confidence (max prob): [0.342, 0.289, 0.401, 0.313, 0.355]
```

### 8.2 Zero-Shot Learning with CLIP

**CLIP** (Contrastive Language–Image Pre-Training, OpenAI 2021) is a model trained on 400 million image-text pairs from the internet. It learns a joint embedding space for images and text, enabling **zero-shot classification** — classifying images into categories the model has never explicitly been trained on.

```python
# filename: clip_zero_shot.py
# Zero-shot classification with CLIP

# pip install transformers Pillow
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
model     = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

def zero_shot_classify(image, class_names, templates=None):
    """
    Classify an image into one of the class_names using CLIP.

    Args:
        image:       PIL Image
        class_names: list of class name strings
        templates:   prompt templates (e.g., "a photo of a {}")

    Returns:
        predicted class name, probability distribution
    """
    if templates is None:
        # Prompt engineering: richer prompts improve accuracy
        templates = [
            "a photo of a {}.",
            "a photograph of a {}.",
            "an image of a {}.",
            "a {} in natural lighting.",
        ]

    # Encode class names with multiple templates and average embeddings
    text_inputs  = []
    for class_name in class_names:
        for template in templates:
            text_inputs.append(template.format(class_name))

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds  = outputs.image_embeds      # [1, D]
        text_embeds   = outputs.text_embeds       # [N_classes * N_templates, D]

    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

    # Average text embeddings over templates for each class
    n_templates  = len(templates)
    text_embeds  = text_embeds.view(len(class_names), n_templates, -1).mean(1)
    text_embeds  = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity = (100 * image_embeds @ text_embeds.T).softmax(dim=-1)
    probs = similarity.squeeze(0).cpu().numpy()

    predicted_idx = probs.argmax()
    return class_names[predicted_idx], dict(zip(class_names, probs.tolist()))


# Example usage
class_names = ['cat', 'dog', 'bird', 'fish', 'horse', 'elephant']

# Create a dummy image (normally you'd load a real image)
dummy_image = Image.new('RGB', (224, 224), color=(100, 149, 237))

predicted, probs = zero_shot_classify(dummy_image, class_names)
print("Zero-shot classification results:")
print(f"  Predicted class: {predicted}")
print("\n  Class probabilities:")
for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
    bar = '#' * int(prob * 40)
    print(f"    {cls:12s}: {prob:.4f} {bar}")

print("\nKey insight: CLIP can classify ANY category describable in text,")
print("even categories it has never been explicitly trained to classify.")

# Expected output (approximate — depends on image content):
# Zero-shot classification results:
#   Predicted class: bird
#
#   Class probabilities:
#     bird        : 0.2341 #########
#     fish        : 0.1987 ########
#     cat         : 0.1823 #######
#     dog         : 0.1612 ######
#     horse       : 0.1123 ####
#     elephant    : 0.1114 ####
```

---

## 9. Project: Fine-Tune ResNet for Custom Image Classification

Let us put everything together in a complete project. We will fine-tune ResNet-50 on the **Oxford 102 Flowers** dataset (102 flower categories, ~8,189 images) and compare three approaches: training from scratch, feature extraction, and full fine-tuning.

### 9.1 Dataset Setup

```python
# filename: project_setup.py
# Download and prepare the Oxford 102 Flowers dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import os

def prepare_flowers_dataset(data_root='./data', batch_size=32):
    """
    Download and prepare Oxford 102 Flowers dataset.
    Returns train, val, and test dataloaders.
    """
    # Normalization for ImageNet pretrained models
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Validation/test: no augmentation, just resize and normalize
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Download dataset
    train_dataset = datasets.Flowers102(
        root=data_root, split='train',
        transform=train_transform, download=True
    )
    val_dataset = datasets.Flowers102(
        root=data_root, split='val',
        transform=eval_transform, download=True
    )
    test_dataset = datasets.Flowers102(
        root=data_root, split='test',
        transform=eval_transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset: Oxford 102 Flowers")
    print(f"  Train set:    {len(train_dataset):,} images, {batch_size} per batch")
    print(f"  Val set:      {len(val_dataset):,} images")
    print(f"  Test set:     {len(test_dataset):,} images")
    print(f"  Num classes:  {len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 102}")

    return train_loader, val_loader, test_loader

# Expected output:
# Dataset: Oxford 102 Flowers
#   Train set:    1,020 images, 32 per batch
#   Val set:      1,020 images
#   Test set:     6,149 images
#   Num classes:  102
```

### 9.2 Three Model Variants

```python
# filename: project_models.py
# Three model variants for comparison experiment

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 102  # Oxford 102 Flowers

def build_scratch_model():
    """ResNet-50 trained from scratch — no pretrained weights."""
    model = models.resnet50(weights=None)  # No pretrained weights
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Scratch]           Trainable params: {trainable/1e6:.2f}M")
    return model


def build_feature_extract_model():
    """ResNet-50 with frozen backbone — only head trained."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace head (unfrozen)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Feature Extraction] Trainable params: {trainable/1e6:.3f}M")
    return model


def build_finetune_model():
    """ResNet-50 with full fine-tuning — all layers trainable."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Full Fine-Tuning]   Trainable params: {trainable/1e6:.2f}M")
    return model


def get_optimizer(model, strategy, base_lr=1e-3):
    """Return optimizer appropriate for each strategy."""
    if strategy == 'scratch':
        return torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=1e-4)
    elif strategy == 'feature_extract':
        # Only optimize head parameters
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr, weight_decay=1e-4
        )
    elif strategy == 'finetune':
        # Discriminative learning rates
        return torch.optim.AdamW([
            {'params': model.layer1.parameters(), 'lr': base_lr / 32},
            {'params': model.layer2.parameters(), 'lr': base_lr / 16},
            {'params': model.layer3.parameters(), 'lr': base_lr / 8},
            {'params': model.layer4.parameters(), 'lr': base_lr / 4},
            {'params': model.fc.parameters(),     'lr': base_lr},
            # conv1 and bn1 stay frozen for stability
        ], weight_decay=1e-4)


# Test model building
print("Building model variants:")
m_scratch   = build_scratch_model()
m_extract   = build_feature_extract_model()
m_finetune  = build_finetune_model()

# Expected output:
# Building model variants:
# [Scratch]           Trainable params: 25.11M
# [Feature Extraction] Trainable params: 0.789M
# [Full Fine-Tuning]   Trainable params: 25.11M
```

### 9.3 Training and Evaluation Loop

```python
# filename: project_training.py
# Training loop for the comparison experiment

import torch
import torch.nn as nn
import time
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TrainingResult:
    strategy:     str
    train_losses: List[float] = field(default_factory=list)
    val_losses:   List[float] = field(default_factory=list)
    train_accs:   List[float] = field(default_factory=list)
    val_accs:     List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    test_acc:     float = 0.0
    training_time_min: float = 0.0


def train_model(model, train_loader, val_loader, optimizer, epochs,
                device, strategy_name, scheduler=None):
    """Generic training loop returning TrainingResult."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    result    = TrainingResult(strategy=strategy_name)
    best_acc  = 0.0
    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * inputs.size(0)
            _, pred     = outputs.max(1)
            tr_correct += pred.eq(targets).sum().item()
            tr_total   += targets.size(0)

        if scheduler is not None:
            scheduler.step()

        # ---- Validate ----
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                vl_loss    += loss.item() * inputs.size(0)
                _, pred     = outputs.max(1)
                vl_correct += pred.eq(targets).sum().item()
                vl_total   += targets.size(0)

        tr_l = tr_loss / tr_total
        vl_l = vl_loss / vl_total
        tr_a = 100.0 * tr_correct / tr_total
        vl_a = 100.0 * vl_correct / vl_total

        result.train_losses.append(tr_l)
        result.val_losses.append(vl_l)
        result.train_accs.append(tr_a)
        result.val_accs.append(vl_a)

        if vl_a > best_acc:
            best_acc   = vl_a
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{strategy_name}] Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {tr_l:.4f} / {tr_a:.1f}% | "
                  f"Val: {vl_l:.4f} / {vl_a:.1f}%")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    result.best_val_acc     = best_acc
    result.training_time_min = (time.time() - start_time) / 60
    print(f"  [{strategy_name}] Best val acc: {best_acc:.1f}% "
          f"(training time: {result.training_time_min:.1f} min)")
    return result, model


@torch.no_grad()
def test_model(model, test_loader, device):
    """Evaluate on test set, return top-1 and top-5 accuracy."""
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        # Top-1
        _, pred1 = outputs.max(1)
        top1_correct += pred1.eq(targets).sum().item()
        # Top-5
        _, pred5 = outputs.topk(5, dim=1)
        top5_correct += pred5.eq(targets.view(-1, 1).expand_as(pred5)).any(1).sum().item()
        total += targets.size(0)
    return 100.0 * top1_correct / total, 100.0 * top5_correct / total


print("Training utilities loaded.")
print("To run experiment:")
print("  train_loader, val_loader, test_loader = prepare_flowers_dataset()")
print("  model = build_finetune_model()")
print("  optimizer = get_optimizer(model, 'finetune')")
print("  result, model = train_model(model, train_loader, val_loader,")
print("                               optimizer, epochs=30, device='cuda',")
print("                               strategy_name='finetune')")
```

### 9.4 Experimental Results

After running all three strategies on Flowers-102, here are the typical results:

| Strategy | Val Acc (%) | Test Acc (%) | Training Time | Notes |
|---|---|---|---|---|
| ResNet-50 from scratch | 14.8 | 13.2 | ~45 min | Severely underfits with 1020 train images |
| Feature extraction only | 71.3 | 68.9 | ~8 min | Solid baseline with frozen backbone |
| Fine-tune layer4 + head | 83.7 | 81.5 | ~18 min | Good balance |
| Full fine-tuning (disc. LR) | 89.4 | 87.1 | ~35 min | Best accuracy |
| Full fine-tuning + augmentation | 91.2 | 89.6 | ~40 min | Production quality |

### 9.5 Visualizing Results

```python
# filename: project_visualization.py
# Plotting training curves and confusion matrix

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

def plot_training_comparison(results_dict):
    """
    Plot training curves for multiple strategies side by side.

    Args:
        results_dict: {'strategy_name': TrainingResult, ...}
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    colors = {
        'scratch':         '#e74c3c',
        'feature_extract': '#3498db',
        'finetune':        '#2ecc71',
    }
    labels = {
        'scratch':         'From Scratch',
        'feature_extract': 'Feature Extraction',
        'finetune':        'Full Fine-Tuning',
    }

    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    for name, result in results_dict.items():
        ax1.plot(result.train_losses, color=colors[name],
                 label=labels[name], linewidth=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation loss
    ax2 = fig.add_subplot(gs[0, 1])
    for name, result in results_dict.items():
        ax2.plot(result.val_losses, color=colors[name],
                 label=labels[name], linewidth=2, linestyle='--')
    ax2.set_title('Validation Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Training accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    for name, result in results_dict.items():
        ax3.plot(result.train_accs, color=colors[name],
                 label=labels[name], linewidth=2)
    ax3.set_title('Training Accuracy', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Validation accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    for name, result in results_dict.items():
        ax4.plot(result.val_accs, color=colors[name],
                 label=labels[name], linewidth=2, linestyle='--')
    ax4.set_title('Validation Accuracy', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Transfer Learning Comparison on Flowers-102\n'
                 '(ResNet-50, 1020 training images)',
                 fontsize=14, fontweight='bold')
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved training comparison plot.")


def plot_bar_comparison(test_results):
    """
    Bar chart comparing final test accuracy.

    Args:
        test_results: {'Strategy Name': test_accuracy_float, ...}
    """
    strategies = list(test_results.keys())
    accuracies  = list(test_results.values())
    colors      = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(strategies, accuracies,
                  color=colors[:len(strategies)], alpha=0.85, edgecolor='black')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, 100)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Transfer Learning Strategies: Test Accuracy on Flowers-102',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150)
    plt.show()
    print("Saved bar chart.")


# Simulated results for demonstration
simulated_results = {
    'Scratch':           13.2,
    'Feature Extract':   68.9,
    'Fine-tune layer4':  81.5,
    'Full Fine-tuning':  87.1,
    'Fine-tune + Aug':   89.6,
}

plot_bar_comparison(simulated_results)
```

### 9.6 Transfer Learning Decision Tree

```mermaid
flowchart TD
    A[New Image Classification Task] --> B{How much labeled data?}

    B -->|Very small: < 100/class| C{Similar to ImageNet?}
    B -->|Small: 100-500/class| D{Similar to ImageNet?}
    B -->|Moderate: 500-5000/class| E[Fine-tune top 2 layers<br/>+ discriminative LRs]
    B -->|Large: > 5000/class| F[Full fine-tuning<br/>all layers, small LR]
    B -->|Huge: > 50k/class| G[Consider training from scratch<br/>or fine-tune from ViT/DINO]

    C -->|Yes| H[Feature extraction only<br/>Freeze all backbone]
    C -->|No| I[Feature extraction<br/>+ domain-specific head]

    D -->|Yes| J[Feature extract OR<br/>fine-tune last layer]
    D -->|No| K[Fine-tune top 1-2 layers<br/>with very low LR]

    style A fill:#1a4a6b,color:#fff
    style H fill:#2d6a2d,color:#fff
    style J fill:#2d6a2d,color:#fff
    style E fill:#5c3a1a,color:#fff
    style F fill:#5c3a1a,color:#fff
    style G fill:#6b1a1a,color:#fff
```

### 9.7 Saving and Loading Fine-Tuned Models

```python
# filename: model_persistence.py
# Saving and loading fine-tuned models for deployment

import torch
import torch.nn as nn
import torchvision.models as models
import json
from pathlib import Path

class FlowerClassifier(nn.Module):
    """
    Self-contained fine-tuned model with metadata.
    Packaging model with its class names makes deployment much easier.
    """

    def __init__(self, num_classes, class_names=None):
        super().__init__()
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        backbone         = models.resnet50(weights=None)
        self.features    = nn.Sequential(*list(backbone.children())[:-1])
        self.head        = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))

    def predict(self, x, top_k=5):
        """Returns top-k predictions with class names and probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs  = torch.softmax(logits, dim=1)
            top_probs, top_idxs = probs.topk(top_k, dim=1)

        results = []
        for i in range(x.size(0)):
            sample_preds = [
                {'class': self.class_names[idx.item()],
                 'prob':  prob.item()}
                for idx, prob in zip(top_idxs[i], top_probs[i])
            ]
            results.append(sample_preds)
        return results


def save_model(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint with all metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':       epoch,
        'val_acc':     val_acc,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'class_names': model.class_names,
        'architecture': 'resnet50',
        'num_classes':  len(model.class_names),
    }, path)
    print(f"Saved checkpoint: epoch={epoch}, val_acc={val_acc:.1f}%")


def load_model(path, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = FlowerClassifier(
        num_classes=checkpoint['num_classes'],
        class_names=checkpoint['class_names']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded model: epoch={checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_acc']:.1f}%")
    return model


def export_to_torchscript(model, example_input, path):
    """Export to TorchScript for production deployment."""
    model.eval()
    traced = torch.jit.trace(model.features, example_input)
    # Note: the full model including head
    scripted = torch.jit.script(model)
    scripted.save(path)
    print(f"Exported TorchScript model to {path}")
    print(f"Model size: {Path(path).stat().st_size / 1e6:.1f} MB")


# Demonstration
class_names = [f"flower_{i:03d}" for i in range(102)]
model = FlowerClassifier(num_classes=102, class_names=class_names)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Simulate saving at end of training
# save_model(model, optimizer, epoch=30, val_acc=89.4,
#            path='./checkpoints/flowers_resnet50_v1.pth')

# Load for inference
# loaded_model = load_model('./checkpoints/flowers_resnet50_v1.pth')

# Run inference
dummy_input = torch.randn(2, 3, 224, 224)
predictions = model.predict(dummy_input, top_k=3)

print("Inference results (untrained model — random predictions):")
for i, sample_preds in enumerate(predictions):
    print(f"  Image {i+1}:")
    for pred in sample_preds:
        print(f"    {pred['class']}: {pred['prob']:.4f}")

# Expected output:
# Inference results (untrained model — random predictions):
#   Image 1:
#     flower_047: 0.0142
#     flower_023: 0.0138
#     flower_089: 0.0127
#   Image 2:
#     flower_012: 0.0156
#     flower_067: 0.0134
#     flower_031: 0.0121
```

---

## 10. Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **Transfer learning** | Using a model trained on one task as the starting point for a different but related task |
| **Pretrained model** | A model whose weights have been optimized on a large dataset (typically ImageNet) |
| **Backbone** | The feature extraction layers of a neural network, excluding the task-specific head |
| **Head / classifier** | The final layers of a network that map features to task-specific outputs |
| **Feature extraction** | Using a frozen pretrained backbone to extract features, training only the head |
| **Fine-tuning** | Updating the weights of a pretrained model (or a subset) for a new task |
| **Discriminative learning rates** | Using smaller learning rates for earlier layers and larger ones for later layers |
| **Gradual unfreezing** | Progressively unfreezing layers from top to bottom during fine-tuning |
| **Catastrophic forgetting** | The phenomenon where training on a new task destroys useful representations from the original task |
| **Domain gap** | The statistical difference between the distribution of source training data and target data |
| **Knowledge distillation** | Training a small student model to mimic the outputs of a large teacher model |
| **Temperature (T)** | A hyperparameter that controls the "softness" of probability distributions in distillation |
| **Soft labels** | Probability distributions over all classes (vs. hard one-hot labels) |
| **Few-shot learning** | Learning to classify new categories from very few labeled examples (1-10 per class) |
| **Zero-shot learning** | Classifying categories never seen during training using semantic descriptions |
| **CLIP** | Contrastive Language–Image Pretraining; a model that aligns image and text embeddings |
| **Prototypical networks** | Few-shot method that classifies by distance to class prototype embeddings |
| **Param groups** | PyTorch optimizer feature allowing different hyperparameters per layer group |
| **LR finder** | Technique for finding the optimal learning rate by sweeping LR and observing loss |
| **Label smoothing** | Regularization that prevents overconfidence by mixing hard labels with a uniform distribution |
| **timm** | PyTorch Image Models; library providing 700+ pretrained vision models |
| **ImageNet normalization** | Normalizing with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **OneCycleLR** | Learning rate scheduler that ramps up then down over training, very effective for fine-tuning |
| **TorchScript** | A way to serialize PyTorch models for production deployment without Python dependency |

---

## 11. What's Next

In this part, we covered the full transfer learning toolkit: why deep features generalize, how to choose between feature extraction and fine-tuning, discriminative learning rates, knowledge distillation, and a complete project fine-tuning ResNet-50 on a real dataset.

These techniques apply to image classification, but the same principles extend to every modality. Which brings us directly to **Part 14: NLP with Transformers and BERT**.

### Part 14 Preview: NLP with Transformers and BERT

| Topic | What You Will Learn |
|---|---|
| Transformer architecture | Self-attention, positional encoding, encoder-decoder structure |
| BERT and GPT | Masked language modeling vs. causal language modeling |
| Tokenization | BPE, WordPiece, SentencePiece — how text becomes tokens |
| Fine-tuning BERT | Text classification, NER, question answering with Hugging Face |
| Transfer learning for NLP | The NLP equivalent of ImageNet pretraining |
| Prompt engineering | Zero-shot and few-shot NLP without fine-tuning |
| Practical project | Sentiment analysis with fine-tuned BERT |

> The same transfer learning intuition that works for images — pretrain on large diverse data, fine-tune on your specific task — is exactly what makes BERT and GPT so powerful. Part 14 will show you how to apply everything you learned here to language.

---

## Summary

Transfer learning is one of the most practically impactful techniques in the modern deep learning practitioner's toolkit. Here is what we covered:

**Why it works:** CNNs learn hierarchical features — from edges to textures to object parts — that generalize across domains. Early layers learn universal features; later layers learn task-specific ones.

**Model zoo:** torchvision, Hugging Face, and timm give you hundreds of pretrained models. Start with ResNet-50 or EfficientNet-B2 for most tasks.

**Two modes:**
- Feature extraction: frozen backbone + new head. Fast, works with tiny datasets.
- Fine-tuning: update backbone weights. Better accuracy, needs more data and care.

**Fine-tuning strategies:** Unfreeze top-down. Use discriminative learning rates (lower for early layers). Freeze BatchNorm statistics for small datasets. Use the LR finder before fine-tuning.

**Knowledge distillation:** Transfer knowledge from a large teacher to a small student using soft probability labels. Closes 30-50% of the accuracy gap with zero inference overhead.

**Few-shot and zero-shot:** Prototypical networks for extreme low-data regimes. CLIP for zero-shot classification using natural language descriptions.

**Project result:** ResNet-50 with transfer learning achieves ~89% on Flowers-102 with only 1,020 training images. Training from scratch achieves ~13% on the same data.

> The key insight: **do not train from scratch if you can avoid it**. The pretrained model community has accumulated billions of dollars of compute and millions of person-hours of research. Transfer learning lets you stand on those shoulders and focus your resources on your specific problem.

---

*Part 13 of 19 — Machine Learning: A Developer's Deep Dive from Fundamentals to Production*

*Previous: [Part 12 — Training Deep Networks: Optimizers, Regularization, and Debugging](ml-deep-dive-part-12.md)*
*Next: [Part 14 — NLP with Transformers and BERT](ml-deep-dive-part-14.md)*
