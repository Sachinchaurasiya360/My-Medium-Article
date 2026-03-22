# Machine Learning Deep Dive — Part 18: MLOps — From Notebook to Production Pipeline

---

**Series:** Machine Learning — A Developer's Deep Dive from Fundamentals to Production
**Part:** 18 of 19 (Production ML)
**Audience:** Developers with Python experience who want to master machine learning from the ground up
**Reading time:** ~60 minutes

---

## Recap: Where We Left Off

In Part 17 we tackled ML system design — how to architect scalable systems with feature stores, real-time versus batch serving trade-offs, and A/B testing frameworks that let you safely roll out new models to subsets of traffic. You saw how decisions made at the architecture level determine whether a model scales to millions of requests or falls over under load.

You have a great model and a system design. Now how do you ship it reliably, repeatedly, and safely? **MLOps — Machine Learning Operations** — is the practice of treating ML systems like software systems: with version control, automated testing, CI/CD pipelines, and continuous monitoring. It's the difference between a model that lives forever in a notebook and one that runs in production for years.

---

## Table of Contents

1. [What is MLOps and Why It Matters](#1-what-is-mlops-and-why-it-matters)
2. [Experiment Tracking with MLflow](#2-experiment-tracking-with-mlflow)
3. [Weights & Biases (W&B)](#3-weights--biases-wb)
4. [Model Registry and Versioning](#4-model-registry-and-versioning)
5. [Data Versioning with DVC](#5-data-versioning-with-dvc)
6. [CI/CD for ML](#6-cicd-for-ml)
7. [Containerization for ML](#7-containerization-for-ml)
8. [Model Serving with FastAPI](#8-model-serving-with-fastapi)
9. [ONNX Export for Deployment](#9-onnx-export-for-deployment)
10. [Monitoring in Production](#10-monitoring-in-production)
11. [Automated Retraining Pipeline](#11-automated-retraining-pipeline)
12. [Project: Full MLOps Pipeline](#12-project-full-mlops-pipeline)
13. [Vocabulary Cheat Sheet](#vocabulary-cheat-sheet)
14. [What's Next](#whats-next)

---

## 1. What is MLOps and Why It Matters

Software engineering has DevOps. Machine learning has **MLOps**.

The term blends "Machine Learning" with "Operations" and refers to the set of practices, tools, and cultural norms that bring the discipline of software engineering to the ML lifecycle. The core insight is simple but powerful: a model that never makes it to production — or that degrades silently once it does — has zero business value.

### The ML Lifecycle

Every production ML system goes through the same fundamental cycle:

```mermaid
flowchart LR
    A[Data Collection\n& Labeling] --> B[Feature\nEngineering]
    B --> C[Experiment\n& Train]
    C --> D[Evaluate\n& Validate]
    D --> E{Good\nEnough?}
    E -- No --> C
    E -- Yes --> F[Register\nModel]
    F --> G[Deploy\nto Production]
    G --> H[Monitor\n& Observe]
    H --> I{Drift or\nDegradation?}
    I -- No --> H
    I -- Yes --> A
```

Each arrow in that diagram represents a hand-off that can fail. MLOps is the discipline of making each hand-off reliable, auditable, and automatable.

### MLOps Maturity Levels

Google's MLOps white paper defines three maturity levels. Most teams start at level 0 and should aim for level 2.

| Level | Name | Description | Manual Steps | Automation |
|-------|------|-------------|-------------|------------|
| **0** | Manual Process | Scripts in notebooks, manual model deployment, no CI/CD | Everything | None |
| **1** | ML Pipeline Automation | Training pipeline is automated, but CI/CD for the pipeline itself is manual | Model deployment, pipeline changes | Training pipeline |
| **2** | CI/CD Pipeline Automation | Full CI/CD for both ML pipeline and serving system | None (approvals only) | Everything including pipeline updates |

> **The gap between Level 0 and Level 1 is where most teams live.** They have training scripts but no automated retraining, no drift detection, and no deployment gates. Getting from Level 0 to Level 1 delivers most of the business value.

### The Three Axes of MLOps

MLOps practitioners track three distinct versioning concerns:

| Axis | What Changes | Tools |
|------|-------------|-------|
| **Code** | Training scripts, serving code, pipelines | Git, GitHub Actions |
| **Data** | Training datasets, feature definitions, label schema | DVC, Delta Lake, LakeFS |
| **Model** | Weights, hyperparameters, framework version | MLflow, W&B, SageMaker |

All three must be versioned in lockstep. If you retrain a model six months from now you need to reproduce: exactly which data was used, exactly which code ran, and exactly which hyperparameters were chosen.

### MLOps vs DevOps: What's Different

| Dimension | DevOps | MLOps |
|-----------|--------|-------|
| **Artifact** | Binary / container image | Model weights + metadata |
| **Testing** | Unit, integration, end-to-end | + Statistical tests on data and predictions |
| **Failure mode** | System crash (obvious) | Silent degradation (insidious) |
| **Trigger for deployment** | Code change | Code change OR data change OR time |
| **Rollback** | Deploy previous version | Deploy previous model version |
| **Reproducibility** | Code + config | Code + config + data + random seed |

The most dangerous failure mode in ML is **silent degradation** — the model keeps returning HTTP 200 but its accuracy has fallen off a cliff because the world changed. That never happens in traditional software.

---

## 2. Experiment Tracking with MLflow

The most common question after a long hyperparameter search: "Which run gave us that 94% accuracy last Tuesday?" Without experiment tracking, the answer is "I'm not sure."

**MLflow** is an open-source platform for the ML lifecycle. It is framework-agnostic, runs locally or on a remote server, and integrates with virtually every major ML library.

### MLflow Components

| Component | Purpose |
|-----------|---------|
| **MLflow Tracking** | Log parameters, metrics, and artifacts per run |
| **MLflow Models** | Standard model packaging format |
| **MLflow Model Registry** | Central store for model versions with staging |
| **MLflow Projects** | Reproducible code packaging |

### Installation and Setup

```python
# requirements.txt
mlflow>=2.10.0
scikit-learn>=1.4.0
torch>=2.2.0
```

```bash
# Start tracking server locally
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

### Basic Tracking: sklearn Integration

```python
# train_sklearn_mlflow.py
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Point to your tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-classification")

def train_model(n_estimators: int, max_depth: int, min_samples_split: int):
    """Train a RandomForest and track everything with MLflow."""

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run(run_name=f"rf-{n_estimators}-depth{max_depth}"):

        # --- Log hyperparameters ---
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "scaler": "StandardScaler",
            "random_state": 42,
        })

        # --- Train ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # --- Log metrics ---
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": auc,
        })

        # --- Log feature importances as artifact ---
        import pandas as pd
        importance_df = pd.DataFrame({
            "feature": data.feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        # --- Log the model ---
        # MLflow wraps sklearn models and saves the scaler + model together
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train_scaled, y_pred)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_scaled[:5],
            registered_model_name="breast-cancer-rf",
        )

        print(f"Run complete — accuracy={accuracy:.4f}, AUC={auc:.4f}")
        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    # Grid search over hyperparameters, all tracked automatically
    configs = [
        (100, 5, 2),
        (200, 10, 5),
        (300, None, 2),
        (100, 10, 10),
    ]

    best_run_id = None
    best_auc = 0.0

    for n_est, depth, min_split in configs:
        run_id = train_model(n_est, depth, min_split)

        # Check if this was the best run
        run = mlflow.get_run(run_id)
        auc = run.data.metrics["roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_run_id = run_id

    print(f"\nBest run: {best_run_id} with AUC={best_auc:.4f}")
```

### PyTorch Training Loop with MLflow

```python
# train_pytorch_mlflow.py
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-pytorch")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_pytorch(lr: float, hidden_dims: list, dropout: float, epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = MLP(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_name = f"mlp-lr{lr}-{'x'.join(map(str, hidden_dims))}"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "lr": lr,
            "hidden_dims": str(hidden_dims),
            "dropout": dropout,
            "epochs": epochs,
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "device": str(device),
        })

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_ds)
            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item() * len(X_batch)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == y_batch).sum().item()

            val_loss /= len(val_ds)
            val_acc = correct / len(val_ds)

            # Log metrics every epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model checkpoint as artifact
                torch.save(model.state_dict(), "best_model.pt")
                mlflow.log_artifact("best_model.pt")

        # Log the final model with signature
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        example_input = torch.tensor(X_val[:5], dtype=torch.float32).to(device)
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            registered_model_name="breast-cancer-pytorch",
        )

        print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    configs = [
        (1e-3, [128, 64, 32], 0.3),
        (5e-4, [256, 128, 64], 0.2),
        (1e-3, [64, 32], 0.4),
    ]
    for lr, hidden, dropout in configs:
        train_pytorch(lr, hidden, dropout)
```

### MLflow Tracking Flow

```mermaid
flowchart TD
    A[Training Script] -->|mlflow.start_run| B[Active Run]
    B -->|log_params| C[(MLflow\nTracking Store\nSQLite / PostgreSQL)]
    B -->|log_metrics| C
    B -->|log_artifact| D[(Artifact Store\nLocal / S3 / GCS)]
    B -->|log_model| D
    C --> E[MLflow UI\nlocalhost:5000]
    D --> E
    E --> F[Compare Runs]
    E --> G[Register Best Model]
    G --> H[Model Registry]
```

### Loading a Logged Model

```python
# load_and_predict.py
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")

# Load by run ID
run_id = "abc123def456"
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Or load from registry by name and version
model = mlflow.sklearn.load_model("models:/breast-cancer-rf/Production")

# Predict
X_new = np.array([[...]])  # your features
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

---

## 3. Weights & Biases (W&B)

**Weights & Biases (W&B)** is a commercial (free tier available) MLOps platform with a polished UI, powerful sweep functionality for hyperparameter optimization, and strong team collaboration features.

### Basic Integration

```python
# train_wandb.py
import wandb
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize a run
wandb.init(
    project="breast-cancer-classification",
    name="mlp-experiment-1",
    config={
        "lr": 1e-3,
        "hidden_dims": [128, 64, 32],
        "dropout": 0.3,
        "epochs": 50,
        "batch_size": 32,
        "optimizer": "Adam",
    },
    tags=["pytorch", "mlp", "baseline"],
    notes="First experiment with MLP on breast cancer dataset",
)

# Access config (wandb.config is swept automatically during sweeps)
config = wandb.config

# ... (build model using config.lr, config.hidden_dims, etc.)

# During training loop
for epoch in range(config.epochs):
    train_loss = ...  # your training logic
    val_accuracy = ...

    # Log metrics — shows up in W&B dashboard in real time
    wandb.log({
        "train/loss": train_loss,
        "val/accuracy": val_accuracy,
        "epoch": epoch,
    })

# Watch model gradients and parameters
model = nn.Linear(30, 1)  # simplified
wandb.watch(model, log="all", log_freq=10)

# Log a custom plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [0.8, 0.85, 0.9])
wandb.log({"learning_curve": wandb.Image(fig)})
plt.close()

wandb.finish()
```

### W&B Sweeps: Automated Hyperparameter Optimization

```python
# sweep_config.py — define the sweep
sweep_config = {
    "method": "bayes",  # bayesian optimization (also: grid, random)
    "metric": {
        "name": "val/accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "hidden_dim_1": {"values": [64, 128, 256]},
        "hidden_dim_2": {"values": [32, 64, 128]},
        "dropout": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.5,
        },
        "batch_size": {"values": [16, 32, 64]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
    },
}

import wandb


def train_sweep():
    """Called by wandb agent for each trial."""
    with wandb.init() as run:
        config = run.config

        # Build model from sweep config
        # ... training code here ...

        # W&B automatically logs metrics and the sweep agent
        # picks the next hyperparameter combination to try
        pass


# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="breast-cancer-classification")

# Run the sweep agent (can run on multiple machines in parallel)
wandb.agent(sweep_id, function=train_sweep, count=20)
```

### W&B Artifacts for Data and Model Versioning

```python
# artifact_versioning.py
import wandb
import pandas as pd

# --- Save a dataset as an artifact ---
with wandb.init(project="breast-cancer-classification", job_type="data-prep") as run:
    artifact = wandb.Artifact(
        name="breast-cancer-dataset",
        type="dataset",
        description="Preprocessed breast cancer dataset v2",
        metadata={"n_samples": 569, "n_features": 30, "split": "train/test 80/20"},
    )

    # Add files to artifact
    artifact.add_file("data/train.csv")
    artifact.add_file("data/test.csv")

    run.log_artifact(artifact)

# --- Save a model as an artifact ---
with wandb.init(project="breast-cancer-classification", job_type="training") as run:

    # Use the dataset artifact as input (creates lineage graph)
    dataset_artifact = run.use_artifact("breast-cancer-dataset:latest")
    dataset_dir = dataset_artifact.download()

    # ... train model ...

    model_artifact = wandb.Artifact(
        name="breast-cancer-model",
        type="model",
        metadata={
            "accuracy": 0.9473,
            "f1_score": 0.9412,
            "framework": "pytorch",
        },
    )
    model_artifact.add_file("best_model.pt")
    run.log_artifact(model_artifact)
```

### MLflow vs W&B Comparison

| Feature | MLflow | Weights & Biases |
|---------|--------|-----------------|
| **Cost** | Free, self-hosted | Free tier + paid plans |
| **Setup** | Run your own server | Cloud-hosted (SaaS) |
| **UI quality** | Good | Excellent |
| **Hyperparameter sweeps** | Manual (Optuna integration) | Built-in Bayesian sweeps |
| **Collaboration** | Limited | Strong (teams, reports) |
| **Model registry** | Built-in | Built-in |
| **Data versioning** | Artifacts (limited) | Artifacts + lineage |
| **On-premise** | Yes (full control) | Enterprise only |
| **Integrations** | Extensive | Extensive |

> **Rule of thumb:** Use MLflow when you need full control and on-premise hosting. Use W&B when team collaboration and a polished UI matter more than self-hosting.

---

## 4. Model Registry and Versioning

A **model registry** is a centralized catalog of trained model versions. It answers: "Which model is in production right now? What were its metrics? When was it deployed? Who approved it?"

### MLflow Model Registry: Stage Transitions

```mermaid
stateDiagram-v2
    [*] --> None: mlflow.register_model()
    None --> Staging: transition_model_version_stage("Staging")
    Staging --> Production: After validation & approval
    Production --> Archived: When replaced by newer version
    Staging --> Archived: If experiment failed
    Archived --> [*]
```

### Registering and Promoting Models

```python
# model_registry.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# --- Register a model from a completed run ---
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name="breast-cancer-classifier",
    tags={
        "training_data_version": "v2.1",
        "framework": "sklearn",
        "author": "ml-team",
    },
)
print(f"Registered version: {model_version.version}")


# --- Add description and tags to version ---
client.update_model_version(
    name="breast-cancer-classifier",
    version=model_version.version,
    description="RandomForest trained on v2.1 data. AUC=0.9812, F1=0.9714",
)

# --- Transition to Staging ---
client.transition_model_version_stage(
    name="breast-cancer-classifier",
    version=model_version.version,
    stage="Staging",
    archive_existing_versions=False,  # keep existing staging model
)
print(f"Model v{model_version.version} moved to Staging")


# --- Run validation tests on Staging model ---
def validate_staging_model(model_name: str, version: str) -> bool:
    """Return True if model passes all validation gates."""

    # Load staging model
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")

    # Run validation test suite
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np

    data = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Staging model — Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Gates: must exceed thresholds to proceed to Production
    ACCURACY_THRESHOLD = 0.92
    AUC_THRESHOLD = 0.97

    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAIL: Accuracy {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}")
        return False
    if auc < AUC_THRESHOLD:
        print(f"FAIL: AUC {auc:.4f} < threshold {AUC_THRESHOLD}")
        return False

    print("PASS: All validation gates passed")
    return True


# --- Promote to Production if validation passes ---
if validate_staging_model("breast-cancer-classifier", model_version.version):
    client.transition_model_version_stage(
        name="breast-cancer-classifier",
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,  # archive the old production model
    )
    print(f"Model v{model_version.version} promoted to Production!")
else:
    client.transition_model_version_stage(
        name="breast-cancer-classifier",
        version=model_version.version,
        stage="Archived",
    )
    print(f"Model v{model_version.version} archived due to failed validation")


# --- Query the registry ---
def get_production_model(model_name: str):
    """Load the current production model."""
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


def list_all_versions(model_name: str):
    """List all registered versions with their stages."""
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"  v{v.version}: {v.current_stage} | {v.description[:50] if v.description else 'No description'}")
```

### Model Signatures

```python
# model_signatures.py
import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Option 1: Infer signature from data
X_train = np.random.randn(100, 30).astype(np.float32)
y_pred = np.random.randint(0, 2, 100)
signature = infer_signature(X_train, y_pred)

# Option 2: Define schema explicitly
input_schema = Schema([
    ColSpec("double", "mean_radius"),
    ColSpec("double", "mean_texture"),
    ColSpec("double", "mean_perimeter"),
    # ... 27 more features
])
output_schema = Schema([ColSpec("integer", "prediction")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log model with signature — MLflow will validate inputs at serve time
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:3],  # saved with model for documentation
    )
```

---

## 5. Data Versioning with DVC

Git is excellent for code but breaks with large data files. Committing a 10GB dataset to git is a disaster: the repo grows permanently, clones are slow, and diffs are meaningless.

**DVC (Data Version Control)** solves this by storing large files in remote storage (S3, GCS, Azure Blob) and tracking only a small `.dvc` pointer file in git.

### DVC Setup

```bash
# Install DVC
pip install dvc dvc-s3  # or dvc-gs, dvc-azure

# Initialize DVC in your git repo
cd my-ml-project
git init
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"

# Configure remote storage
dvc remote add -d myremote s3://my-ml-bucket/dvc-store
dvc remote modify myremote region us-east-1
git add .dvc/config
git commit -m "Configure DVC remote"
```

### Tracking Data Files

```bash
# Track a large dataset with DVC
dvc add data/raw/breast_cancer.csv
# Creates: data/raw/breast_cancer.csv.dvc  (small pointer file — commit to git)
# Adds:    data/raw/breast_cancer.csv to .gitignore

git add data/raw/breast_cancer.csv.dvc data/raw/.gitignore
git commit -m "Add raw breast cancer dataset v1"
git tag -a "data-v1.0" -m "Initial dataset"

# Push data to remote storage
dvc push  # uploads breast_cancer.csv to S3

# On another machine: pull data
git clone https://github.com/org/my-ml-project
git checkout data-v1.0
dvc pull  # downloads breast_cancer.csv from S3
```

### DVC Pipelines

DVC pipelines define reproducible ML workflows as a directed acyclic graph (DAG).

```yaml
# dvc.yaml — define your ML pipeline
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/breast_cancer.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - prepare.test_size
      - prepare.random_seed

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/processed/train.csv
      - data/processed/test.csv
    outs:
      - data/features/train_features.pkl
      - data/features/test_features.pkl

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features/train_features.pkl
    outs:
      - models/model.pkl
    params:
      - train.n_estimators
      - train.max_depth
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/features/test_features.pkl
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.png
```

```yaml
# params.yaml — pipeline parameters
prepare:
  test_size: 0.2
  random_seed: 42

train:
  n_estimators: 200
  max_depth: 10
  min_samples_split: 5
```

```bash
# Run the full pipeline (only reruns changed stages)
dvc repro

# Compare experiments
dvc params diff
dvc metrics diff

# Show pipeline DAG
dvc dag
```

### DVC Data Lineage

```python
# src/prepare.py — DVC pipeline stage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import yaml
import os

def prepare():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]

    # Load raw data
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_seed"],
        stratify=df["target"],
    )

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

if __name__ == "__main__":
    prepare()
```

---

## 6. CI/CD for ML

CI/CD for ML extends the traditional pipeline with ML-specific test stages. A commit to the training code or a new version of the data should automatically trigger training, evaluation, and (conditionally) deployment.

### What to Test in ML

| Test Type | What It Checks | Example |
|-----------|---------------|---------|
| **Unit tests** | Individual functions in isolation | `test_feature_engineering.py` |
| **Integration tests** | Full pipeline with small data | End-to-end train + predict |
| **Model tests** | Accuracy above baseline, inference speed | AUC > 0.95, latency < 50ms |
| **Data tests** | Schema validation, no nulls, distribution | `great_expectations` or custom |
| **Regression tests** | Predictions haven't changed unexpectedly | Snapshot test on fixed inputs |

### Python Test Suite

```python
# tests/test_pipeline.py
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import time

# Assume your project has a src/ package
from src.features import build_features, validate_feature_schema
from src.train import train_model
from src.predict import predict


class TestFeatureEngineering:
    """Unit tests for feature transformations."""

    def test_build_features_output_shape(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        features = build_features(df)
        assert features.shape[0] == len(df), "Row count should be preserved"
        assert features.shape[1] > 0, "Should produce at least one feature"

    def test_build_features_no_nulls(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        features = build_features(df)
        assert not features.isnull().any().any(), "Features should have no nulls"

    def test_build_features_finite_values(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        features = build_features(df)
        assert np.isfinite(features.values).all(), "All feature values must be finite"

    def test_feature_schema_validation(self):
        """Schema validator should catch wrong column names."""
        bad_df = pd.DataFrame({"wrong_col": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing required features"):
            validate_feature_schema(bad_df)


class TestModel:
    """Model quality and performance tests."""

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Train model once for all tests in this class."""
        data = load_breast_cancer()
        X, y = data.data, data.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_model(X_scaled, y), scaler, X_scaled, y

    def test_accuracy_above_baseline(self, trained_model):
        model, scaler, X, y = trained_model
        from sklearn.metrics import accuracy_score
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        # Model should be significantly better than majority class baseline (~62%)
        assert acc > 0.92, f"Accuracy {acc:.4f} is below minimum threshold 0.92"

    def test_roc_auc_above_threshold(self, trained_model):
        model, scaler, X, y = trained_model
        from sklearn.metrics import roc_auc_score
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        assert auc > 0.97, f"AUC {auc:.4f} is below minimum threshold 0.97"

    def test_inference_latency(self, trained_model):
        """Single prediction must complete in under 100ms."""
        model, scaler, X, y = trained_model
        single_input = X[:1]

        # Warm up
        model.predict(single_input)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            model.predict(single_input)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000

        assert elapsed_ms < 100, f"Inference too slow: {elapsed_ms:.2f}ms"

    def test_batch_inference_latency(self, trained_model):
        """Batch of 1000 predictions must complete in under 500ms."""
        model, scaler, X, y = trained_model
        batch = X[:1000]

        start = time.perf_counter()
        model.predict(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Batch inference too slow: {elapsed_ms:.2f}ms"

    def test_prediction_determinism(self, trained_model):
        """Same input should always produce the same output."""
        model, scaler, X, y = trained_model
        input_data = X[:10]
        pred1 = model.predict(input_data)
        pred2 = model.predict(input_data)
        np.testing.assert_array_equal(pred1, pred2)


class TestDataQuality:
    """Data validation tests."""

    def test_no_target_leakage(self):
        """Features should not contain the target variable."""
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

        features = build_features(df.drop("target", axis=1))

        # Target should not be predictable with R² = 1.0 from features
        # (simple check: target column not in feature columns)
        assert "target" not in features.columns

    def test_class_balance_acceptable(self):
        """Neither class should be < 10% of the dataset."""
        data = load_breast_cancer()
        y = data.target
        class_ratios = np.bincount(y) / len(y)
        assert all(r > 0.10 for r in class_ratios), \
            f"Class imbalance detected: {class_ratios}"
```

### GitHub Actions CI/CD Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - "src/**"
      - "tests/**"
      - "dvc.yaml"
      - "params.yaml"
  pull_request:
    branches: [main]
  workflow_dispatch:  # allow manual trigger

env:
  PYTHON_VERSION: "3.11"
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  # -------------------------------------------------------
  # Job 1: Code quality and unit tests
  # -------------------------------------------------------
  test:
    name: Test & Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov ruff black isort

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Format check with black
        run: black --check src/ tests/

      - name: Run unit tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # -------------------------------------------------------
  # Job 2: Data validation
  # -------------------------------------------------------
  validate-data:
    name: Validate Data
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt dvc dvc-s3

      - name: Configure DVC
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID

      - name: Pull data
        run: dvc pull data/

      - name: Run data validation tests
        run: pytest tests/test_data_quality.py -v

  # -------------------------------------------------------
  # Job 3: Train and evaluate model
  # -------------------------------------------------------
  train:
    name: Train Model
    runs-on: ubuntu-latest
    needs: validate-data
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt dvc dvc-s3

      - name: Pull data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: dvc pull

      - name: Run DVC pipeline
        run: dvc repro

      - name: Check model metrics
        run: |
          python scripts/check_metrics.py \
            --metrics-file metrics/eval_metrics.json \
            --min-accuracy 0.92 \
            --min-auc 0.97

      - name: Register model to MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ env.MLFLOW_TRACKING_URI }}
        run: python scripts/register_model.py

  # -------------------------------------------------------
  # Job 4: Build and push Docker image
  # -------------------------------------------------------
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: train
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            myorg/ml-model:latest
            myorg/ml-model:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # -------------------------------------------------------
  # Job 5: Deploy to staging
  # -------------------------------------------------------
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    environment: staging
    steps:
      - name: Deploy to staging cluster
        env:
          KUBE_CONFIG: ${{ secrets.KUBE_CONFIG_STAGING }}
        run: |
          echo "$KUBE_CONFIG" > kubeconfig.yaml
          kubectl --kubeconfig=kubeconfig.yaml set image \
            deployment/ml-model ml-model=myorg/ml-model:${{ github.sha }}

  # -------------------------------------------------------
  # Job 6: Integration tests on staging
  # -------------------------------------------------------
  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests against staging
        env:
          API_URL: ${{ secrets.STAGING_API_URL }}
        run: pytest tests/integration/ -v --api-url=$API_URL
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ["--maxkb=500"]  # prevent large files in git

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: no-jupyter-output
        name: Check no Jupyter output committed
        entry: python scripts/check_notebook_output.py
        language: python
        files: \.ipynb$

      - id: run-unit-tests
        name: Run fast unit tests
        entry: pytest tests/unit/ -x -q
        language: system
        pass_filenames: false
        stages: [commit]
```

---

## 7. Containerization for ML

**Docker** solves the "it works on my machine" problem that plagues ML: a model trained in one Python environment may behave differently in another due to library version mismatches, CUDA version differences, or OS-level dependencies.

### ML-Specific Dockerfile

```dockerfile
# Dockerfile
# Multi-stage build: build stage for dependencies, runtime stage for serving

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Add user's local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check — Docker/Kubernetes will call this endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use exec form to handle signals properly
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### GPU-Enabled Dockerfile

```dockerfile
# Dockerfile.gpu
# For GPU inference with NVIDIA CUDA

FROM nvidia/cuda:12.3.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Local ML Stack

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ML model serving API
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=breast-cancer-classifier
      - MODEL_STAGE=Production
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models:ro
    depends_on:
      mlflow:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # MLflow tracking server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # PostgreSQL for MLflow backend
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
      - POSTGRES_DB=mlflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - --config.file=/etc/prometheus/prometheus.yml

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

volumes:
  mlflow-artifacts:
  postgres-data:
  grafana-data:
```

---

## 8. Model Serving with FastAPI

The serving layer is where your model meets real traffic. **FastAPI** is the preferred Python framework for ML APIs: it's fast (ASGI, async), generates OpenAPI docs automatically, and integrates naturally with Pydantic for request validation.

### Production FastAPI Serving App

```python
# src/app.py
import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Prometheus metrics
# ----------------------------------------------------------------
REQUEST_COUNT = Counter(
    "ml_request_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
MODEL_VERSION = Gauge(
    "ml_model_version_info",
    "Current model version",
    ["model_name", "version", "stage"],
)

# ----------------------------------------------------------------
# Global model state (loaded once at startup, not per-request)
# ----------------------------------------------------------------
class ModelState:
    model: Any = None
    model_version: str = "unknown"
    model_name: str = "unknown"
    load_time: float = 0.0


model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    logger.info("Loading model...")
    start = time.time()

    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = os.getenv("MODEL_NAME", "breast-cancer-classifier")
        model_stage = os.getenv("MODEL_STAGE", "Production")

        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = f"models:/{model_name}/{model_stage}"
        model_state.model = mlflow.sklearn.load_model(model_uri)
        model_state.model_name = model_name

        # Get version info
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        if versions:
            model_state.model_version = versions[0].version
            MODEL_VERSION.labels(
                model_name=model_name,
                version=model_state.model_version,
                stage=model_stage,
            ).set(1)

        model_state.load_time = time.time() - start
        logger.info(
            f"Model loaded: {model_name} v{model_state.model_version} "
            f"({model_stage}) in {model_state.load_time:.2f}s"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield  # application runs here

    logger.info("Shutting down — releasing model resources")
    model_state.model = None


# ----------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------
app = FastAPI(
    title="ML Model Serving API",
    description="Production ML inference endpoint",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------
# Request / Response schemas
# ----------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Input features for breast cancer classification."""

    mean_radius: float = Field(..., gt=0, description="Mean radius of cell nuclei")
    mean_texture: float = Field(..., gt=0, description="Mean texture of cell nuclei")
    mean_perimeter: float = Field(..., gt=0, description="Mean perimeter of cell nuclei")
    mean_area: float = Field(..., gt=0)
    mean_smoothness: float = Field(..., gt=0, lt=1)
    mean_compactness: float = Field(..., gt=0)
    mean_concavity: float = Field(..., ge=0)
    mean_concave_points: float = Field(..., ge=0)
    mean_symmetry: float = Field(..., gt=0, lt=1)
    mean_fractal_dimension: float = Field(..., gt=0)
    # ... remaining 20 features (abbreviated here)

    @validator("mean_radius")
    def radius_reasonable(cls, v):
        if v > 100:
            raise ValueError("mean_radius exceeds reasonable range (> 100)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "mean_radius": 14.0,
                "mean_texture": 20.5,
                "mean_perimeter": 90.3,
                "mean_area": 600.0,
                "mean_smoothness": 0.1,
                "mean_compactness": 0.13,
                "mean_concavity": 0.09,
                "mean_concave_points": 0.05,
                "mean_symmetry": 0.18,
                "mean_fractal_dimension": 0.063,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=malignant, 1=benign")
    probability_benign: float = Field(..., ge=0, le=1)
    probability_malignant: float = Field(..., ge=0, le=1)
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    instances: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float


# ----------------------------------------------------------------
# Middleware: request timing
# ----------------------------------------------------------------
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------
_start_time = time.time()


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    return HealthResponse(
        status="ok" if model_state.model is not None else "degraded",
        model_loaded=model_state.model is not None,
        model_name=model_state.model_name,
        model_version=model_state.model_version,
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/metrics", tags=["ops"])
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""

    if model_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start = time.perf_counter()

    try:
        # Convert request to numpy array
        # In practice, you'd use a more robust feature extraction function
        features = np.array([[
            request.mean_radius, request.mean_texture,
            request.mean_perimeter, request.mean_area,
            request.mean_smoothness, request.mean_compactness,
            request.mean_concavity, request.mean_concave_points,
            request.mean_symmetry, request.mean_fractal_dimension,
            # ... remaining features ...
        ]])

        # Run inference (synchronous sklearn predict in thread pool)
        # asyncio.to_thread keeps the event loop unblocked
        prediction_array = await asyncio.to_thread(
            model_state.model.predict, features
        )
        proba_array = await asyncio.to_thread(
            model_state.model.predict_proba, features
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Update metrics
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)

        return PredictionResponse(
            prediction=int(prediction_array[0]),
            probability_benign=float(proba_array[0][1]),
            probability_malignant=float(proba_array[0][0]),
            model_version=model_state.model_version,
            latency_ms=latency_ms,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", tags=["inference"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint for high throughput."""

    if model_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start = time.perf_counter()

    # Build feature matrix from all instances
    features = np.array([
        [inst.mean_radius, inst.mean_texture, inst.mean_perimeter,
         inst.mean_area, inst.mean_smoothness, inst.mean_compactness,
         inst.mean_concavity, inst.mean_concave_points, inst.mean_symmetry,
         inst.mean_fractal_dimension]
        for inst in request.instances
    ])

    predictions = await asyncio.to_thread(model_state.model.predict, features)
    probas = await asyncio.to_thread(model_state.model.predict_proba, features)

    latency_ms = (time.perf_counter() - start) * 1000

    REQUEST_COUNT.labels(endpoint="/predict/batch", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(latency_ms / 1000)

    return {
        "predictions": predictions.tolist(),
        "probabilities_benign": probas[:, 1].tolist(),
        "n_instances": len(request.instances),
        "model_version": model_state.model_version,
        "latency_ms": latency_ms,
    }
```

---

## 9. ONNX Export for Deployment

**ONNX (Open Neural Network Exchange)** is an open format for ML models. Exporting to ONNX lets you:
- Run inference with **ONNX Runtime**, which is often 2-5x faster than native PyTorch
- Deploy to environments without a full ML framework installed (C++, mobile, edge)
- Serve models trained in different frameworks through a single inference engine

### Export PyTorch to ONNX

```python
# onnx_export_pytorch.py
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import time

class MLP(nn.Module):
    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_to_onnx(model: nn.Module, output_path: str, input_dim: int = 30):
    """Export a PyTorch model to ONNX format."""
    model.eval()

    # Create dummy input with dynamic batch size
    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # use latest stable opset
        do_constant_folding=True,  # optimize constants
        input_names=["features"],
        output_names=["probability"],
        dynamic_axes={
            "features": {0: "batch_size"},  # batch dimension is dynamic
            "probability": {0: "batch_size"},
        },
        verbose=False,
    )

    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {output_path}")
    print(f"ONNX opset: {onnx_model.opset_import[0].version}")

    return output_path


def benchmark_inference(model_path: str, n_samples: int = 1000, batch_size: int = 32):
    """Compare PyTorch vs ONNX Runtime inference speed."""

    # Load PyTorch model
    pt_model = MLP()
    pt_model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    pt_model.eval()

    # Create ONNX Runtime session
    # Providers: CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider
    ort_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions(),
    )

    # Generate test data
    X = np.random.randn(n_samples, 30).astype(np.float32)

    # --- PyTorch benchmark ---
    X_tensor = torch.tensor(X)
    # Warm up
    with torch.no_grad():
        for i in range(0, 64, batch_size):
            pt_model(X_tensor[i:i+batch_size])

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_tensor[i:i+batch_size]
            _ = pt_model(batch)
    pt_time = (time.perf_counter() - start) * 1000

    # --- ONNX Runtime benchmark ---
    input_name = ort_session.get_inputs()[0].name
    # Warm up
    for i in range(0, 64, batch_size):
        ort_session.run(None, {input_name: X[i:i+batch_size]})

    start = time.perf_counter()
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        _ = ort_session.run(None, {input_name: batch})
    ort_time = (time.perf_counter() - start) * 1000

    speedup = pt_time / ort_time
    print(f"\nInference benchmark ({n_samples} samples, batch_size={batch_size}):")
    print(f"  PyTorch:      {pt_time:.1f}ms")
    print(f"  ONNX Runtime: {ort_time:.1f}ms")
    print(f"  Speedup:      {speedup:.2f}x")

    return pt_time, ort_time


# Export sklearn model to ONNX using skl2onnx
def export_sklearn_to_onnx(sklearn_model, output_path: str, n_features: int = 30):
    """Export sklearn model to ONNX format."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # Define input type
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    # Convert
    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=initial_type,
        options={type(sklearn_model): {"zipmap": False}},  # return arrays not dicts
        target_opset=17,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"sklearn model exported to ONNX: {output_path}")


if __name__ == "__main__":
    # Create and export a sample model
    model = MLP()
    torch.save(model.state_dict(), "model.pt")

    export_to_onnx(model, "model.onnx")
    pt_time, ort_time = benchmark_inference("model.onnx")
```

---

## 10. Monitoring in Production

Once deployed, the work is not done. **Production monitoring** is arguably more important than the initial deployment. Three categories of things can go wrong:

1. **System failures** — API goes down, memory leak, high latency
2. **Data drift** — input data distribution shifts from what the model was trained on
3. **Concept drift** — the relationship between inputs and outputs changes

### What to Monitor

| Metric Category | Specific Metrics | Alert Threshold |
|----------------|-----------------|-----------------|
| **System** | API latency p50/p95/p99, error rate, throughput | p99 > 200ms, error rate > 1% |
| **Model** | Prediction confidence, prediction distribution | Mean confidence drops > 10% |
| **Data drift** | KS statistic, PSI per feature | PSI > 0.2 for any feature |
| **Business** | Downstream outcomes (labels), feedback loop | Accuracy drops > 5% |
| **Infrastructure** | CPU, GPU, memory, disk | CPU > 80%, memory > 90% |

### Monitoring Architecture

```mermaid
flowchart TB
    subgraph Serving
        A[ML API\nFastAPI] -->|predictions + features| B[(Prediction\nLogs DB)]
    end

    subgraph Monitoring
        B --> C[Drift Detector\nScheduled Job]
        B --> D[Metrics Aggregator\nPrometheus]
        C -->|PSI, KS stats| E[(Metrics Store)]
        D --> E
        E --> F[Grafana\nDashboard]
        E --> G{Alert\nEngine}
    end

    subgraph Response
        G -->|drift alert| H[PagerDuty /\nSlack Alert]
        G -->|retrain trigger| I[Retraining\nPipeline]
        I -->|new model| J[Model Registry]
        J -->|after validation| A
    end
```

### Data Drift Detection from Scratch

```python
# monitoring/drift_detector.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class DriftReport:
    """Results of a drift detection run."""
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    is_drifted: bool
    drift_severity: str  # "none", "warning", "critical"


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI measures the shift in distribution between a reference and current dataset.

    Rules of thumb:
      PSI < 0.1:  No significant change
      PSI 0.1-0.2: Moderate change — investigate
      PSI > 0.2:  Significant change — model likely needs retraining
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # handle duplicate percentiles

    if len(breakpoints) < 2:
        return 0.0

    # Count samples in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * n_bins)
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * n_bins)

    # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    ks_alpha: float = 0.05,
    psi_warning_threshold: float = 0.1,
    psi_critical_threshold: float = 0.2,
) -> Dict[str, DriftReport]:
    """
    Detect data drift for all features.

    Args:
        reference_data: Training data (baseline)
        current_data: Recent production data
        ks_alpha: Significance level for KS test
        psi_warning_threshold: PSI threshold for warning
        psi_critical_threshold: PSI threshold for critical alert

    Returns:
        Dict mapping feature name to DriftReport
    """
    reports = {}

    common_features = [c for c in reference_data.columns if c in current_data.columns]

    for feature in common_features:
        ref = reference_data[feature].dropna().values
        cur = current_data[feature].dropna().values

        if len(ref) < 30 or len(cur) < 30:
            warnings.warn(f"Not enough data for drift detection on {feature}")
            continue

        # KS test: two-sample Kolmogorov-Smirnov test
        # H0: both samples drawn from same distribution
        ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)

        # PSI
        psi = compute_psi(ref, cur)

        # Determine severity
        is_drifted = ks_pvalue < ks_alpha or psi > psi_warning_threshold

        if psi > psi_critical_threshold or (ks_pvalue < ks_alpha and ks_stat > 0.2):
            severity = "critical"
        elif psi > psi_warning_threshold or ks_pvalue < ks_alpha:
            severity = "warning"
        else:
            severity = "none"

        reports[feature] = DriftReport(
            feature=feature,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            psi=float(psi),
            is_drifted=is_drifted,
            drift_severity=severity,
        )

    return reports


def summarize_drift_report(reports: Dict[str, DriftReport]) -> dict:
    """Create a summary suitable for logging/alerting."""
    total = len(reports)
    drifted = sum(1 for r in reports.values() if r.is_drifted)
    critical = sum(1 for r in reports.values() if r.drift_severity == "critical")
    warning = sum(1 for r in reports.values() if r.drift_severity == "warning")

    top_drifted = sorted(
        [r for r in reports.values() if r.is_drifted],
        key=lambda r: r.psi,
        reverse=True,
    )[:5]

    return {
        "total_features": total,
        "drifted_features": drifted,
        "drift_rate": drifted / total if total > 0 else 0.0,
        "critical_features": critical,
        "warning_features": warning,
        "top_drifted": [
            {"feature": r.feature, "psi": r.psi, "severity": r.drift_severity}
            for r in top_drifted
        ],
        "overall_drift": "critical" if critical > 0 else ("warning" if warning > 0 else "none"),
    }
```

### Production Monitoring Endpoint

```python
# monitoring/monitor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from drift_detector import detect_drift, summarize_drift_report


class ProductionMonitor:
    """Monitors a production model for data drift and performance degradation."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        model,
        feature_columns: list,
        alert_callback=None,
    ):
        self.reference_data = reference_data
        self.model = model
        self.feature_columns = feature_columns
        self.alert_callback = alert_callback
        self.prediction_log = []

    def log_prediction(self, features: dict, prediction: int, probability: float):
        """Log a single prediction for monitoring."""
        record = {
            **features,
            "_prediction": prediction,
            "_probability": probability,
            "_timestamp": datetime.utcnow().isoformat(),
        }
        self.prediction_log.append(record)

    def run_drift_check(self, window_hours: int = 24) -> dict:
        """
        Run drift detection on recent production data vs reference.
        Returns a summary report.
        """
        if len(self.prediction_log) < 100:
            return {"status": "insufficient_data", "n_predictions": len(self.prediction_log)}

        # Get recent predictions
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [
            p for p in self.prediction_log
            if datetime.fromisoformat(p["_timestamp"]) > cutoff
        ]

        if len(recent) < 50:
            return {"status": "insufficient_recent_data", "n_predictions": len(recent)}

        # Extract features
        current_df = pd.DataFrame(recent)[self.feature_columns]
        ref_df = self.reference_data[self.feature_columns]

        # Run drift detection
        reports = detect_drift(ref_df, current_df)
        summary = summarize_drift_report(reports)

        summary["window_hours"] = window_hours
        summary["n_predictions_analyzed"] = len(recent)
        summary["checked_at"] = datetime.utcnow().isoformat()

        # Fire alert if needed
        if summary["overall_drift"] in ("warning", "critical") and self.alert_callback:
            self.alert_callback(summary)

        return summary

    def compute_prediction_statistics(self, window_hours: int = 24) -> dict:
        """Compute statistics about predictions in the recent window."""
        if not self.prediction_log:
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [
            p for p in self.prediction_log
            if datetime.fromisoformat(p["_timestamp"]) > cutoff
        ]

        if not recent:
            return {}

        probabilities = [p["_probability"] for p in recent]
        predictions = [p["_prediction"] for p in recent]

        return {
            "n_predictions": len(recent),
            "positive_rate": np.mean(predictions),
            "mean_confidence": np.mean(probabilities),
            "low_confidence_rate": np.mean([p < 0.6 for p in probabilities]),
            "window_hours": window_hours,
        }
```

---

## 11. Automated Retraining Pipeline

**Retraining triggers** can be:
- **Scheduled**: "Retrain every Monday at 2am" — simple but may be wasteful or too infrequent
- **Drift-based**: "Retrain when PSI > 0.2 on any critical feature" — responsive to actual change
- **Performance-based**: "Retrain when rolling accuracy drops below 90%" — requires ground truth labels

```mermaid
flowchart TD
    A[Monitoring Service\nRunning Continuously] --> B{Check\nTrigger Conditions}

    B -->|Scheduled trigger| C[Trigger Retraining]
    B -->|Drift detected PSI > 0.2| C
    B -->|Performance degraded| C
    B -->|No trigger| A

    C --> D[Fetch Latest Data\nfrom Feature Store]
    D --> E[Run Training\nPipeline - DVC repro]
    E --> F[Evaluate New Model\nvs Current Production]

    F --> G{New Model\nBetter?}
    G -- Yes + passes gates --> H[Register to Staging]
    G -- No --> I[Alert: Manual\nReview Needed]

    H --> J[Run Validation\nTest Suite]
    J --> K{All Tests\nPass?}
    K -- Yes --> L[Promote to Production]
    K -- No --> M[Archive + Alert]

    L --> N[Update Serving\nConfiguration]
    N --> O[Monitor New Model\nfor 24h]
    O --> P{Any Regression?}
    P -- Yes --> Q[Rollback to\nPrevious Version]
    P -- No --> A
```

### Retraining Orchestrator

```python
# retraining/orchestrator.py
import logging
import subprocess
import json
from datetime import datetime
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class RetrainingOrchestrator:
    """
    Manages the automated retraining lifecycle:
    1. Train new model
    2. Evaluate vs production baseline
    3. Promote if better, rollback if worse
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        model_name: str,
        accuracy_threshold: float = 0.92,
        auc_threshold: float = 0.97,
        min_improvement: float = 0.005,  # new model must be 0.5% better to replace
    ):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
        self.accuracy_threshold = accuracy_threshold
        self.auc_threshold = auc_threshold
        self.min_improvement = min_improvement

    def get_production_metrics(self) -> Optional[dict]:
        """Get metrics of the current production model."""
        try:
            versions = self.client.get_latest_versions(
                self.model_name, stages=["Production"]
            )
            if not versions:
                logger.info("No production model found — new model will be promoted automatically")
                return None

            v = versions[0]
            run = self.client.get_run(v.run_id)
            return {
                "version": v.version,
                "run_id": v.run_id,
                "accuracy": run.data.metrics.get("accuracy"),
                "roc_auc": run.data.metrics.get("roc_auc"),
            }
        except Exception as e:
            logger.error(f"Failed to get production metrics: {e}")
            return None

    def run_training_pipeline(self) -> str:
        """Execute DVC pipeline and return the new run ID."""
        logger.info("Starting DVC pipeline...")

        result = subprocess.run(
            ["dvc", "repro", "--force"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"DVC pipeline failed:\n{result.stderr}")

        # Read the run ID written by the training script
        with open("artifacts/last_run_id.txt") as f:
            run_id = f.read().strip()

        logger.info(f"Training pipeline completed. Run ID: {run_id}")
        return run_id

    def evaluate_new_model(self, run_id: str) -> dict:
        """Get metrics of the newly trained model."""
        run = self.client.get_run(run_id)
        return {
            "run_id": run_id,
            "accuracy": run.data.metrics.get("accuracy"),
            "roc_auc": run.data.metrics.get("roc_auc"),
        }

    def should_promote(self, new_metrics: dict, prod_metrics: Optional[dict]) -> bool:
        """Determine if the new model should replace production."""

        # Always promote if no production model exists
        if prod_metrics is None:
            return True

        # Must meet absolute thresholds
        if new_metrics["accuracy"] < self.accuracy_threshold:
            logger.warning(
                f"New model accuracy {new_metrics['accuracy']:.4f} "
                f"below threshold {self.accuracy_threshold}"
            )
            return False

        if new_metrics["roc_auc"] < self.auc_threshold:
            logger.warning(
                f"New model AUC {new_metrics['roc_auc']:.4f} "
                f"below threshold {self.auc_threshold}"
            )
            return False

        # Must improve over production by at least min_improvement
        auc_improvement = new_metrics["roc_auc"] - prod_metrics["roc_auc"]
        if auc_improvement < -0.001:  # allow tiny regressions due to randomness
            logger.warning(
                f"New model AUC {new_metrics['roc_auc']:.4f} is worse than "
                f"production {prod_metrics['roc_auc']:.4f}"
            )
            return False

        logger.info(
            f"New model qualifies for promotion: "
            f"AUC {new_metrics['roc_auc']:.4f} vs prod {prod_metrics['roc_auc']:.4f} "
            f"(improvement: {auc_improvement:+.4f})"
        )
        return True

    def run(self, trigger_reason: str = "scheduled") -> dict:
        """Execute the full retraining cycle."""
        started_at = datetime.utcnow().isoformat()
        logger.info(f"Retraining triggered: {trigger_reason}")

        # Step 1: Get current production baseline
        prod_metrics = self.get_production_metrics()
        logger.info(f"Production baseline: {prod_metrics}")

        # Step 2: Train new model
        new_run_id = self.run_training_pipeline()
        new_metrics = self.evaluate_new_model(new_run_id)
        logger.info(f"New model metrics: {new_metrics}")

        # Step 3: Decide whether to promote
        if self.should_promote(new_metrics, prod_metrics):
            # Register new version
            model_uri = f"runs:/{new_run_id}/model"
            version = mlflow.register_model(model_uri, self.model_name)

            # Move to staging first
            self.client.transition_model_version_stage(
                self.model_name, version.version, "Staging"
            )

            # Promote to production (archives old production model)
            self.client.transition_model_version_stage(
                self.model_name, version.version, "Production",
                archive_existing_versions=True,
            )

            outcome = "promoted"
            logger.info(f"New model v{version.version} promoted to Production")
        else:
            outcome = "rejected"
            logger.info("New model did NOT qualify for promotion")

        return {
            "trigger": trigger_reason,
            "started_at": started_at,
            "completed_at": datetime.utcnow().isoformat(),
            "new_run_id": new_run_id,
            "new_metrics": new_metrics,
            "production_metrics": prod_metrics,
            "outcome": outcome,
        }
```

---

## 12. Project: Full MLOps Pipeline

Now let's put it all together. This project builds a complete MLOps pipeline for the breast cancer classifier with all components wired up.

### Project Structure

```
ml-ops-project/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── data/
│   └── raw/              # tracked by DVC
├── src/
│   ├── __init__.py
│   ├── prepare.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── monitoring/
│   ├── drift_detector.py
│   └── monitor.py
├── retraining/
│   └── orchestrator.py
├── scripts/
│   ├── check_metrics.py
│   └── register_model.py
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .pre-commit-config.yaml
```

### Main Training Script

```python
# src/train.py — the heart of the pipeline
import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_features():
    train_df = pd.read_parquet("data/features/train_features.parquet")
    test_df = pd.read_parquet("data/features/test_features.parquet")

    feature_cols = [c for c in train_df.columns if c != "target"]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    return X_train, y_train, X_test, y_test, feature_cols


def get_model(params: dict):
    model_type = params["train"]["model_type"]

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"],
            min_samples_split=params["train"]["min_samples_split"],
            random_state=params["train"]["random_seed"],
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"],
            learning_rate=params["train"]["learning_rate"],
            random_state=params["train"]["random_seed"],
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            C=params["train"]["C"],
            max_iter=1000,
            random_state=params["train"]["random_seed"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train():
    params = load_params()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("breast-cancer-full-pipeline")

    X_train, y_train, X_test, y_test, feature_cols = load_features()

    with mlflow.start_run() as run:
        # Log all params
        mlflow.log_params({
            "model_type": params["train"]["model_type"],
            **{f"train_{k}": v for k, v in params["train"].items()},
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(feature_cols),
        })

        # Build and train model
        model = get_model(params)
        model.fit(X_train, y_train)

        # Cross-validation on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mlflow.log_metrics({
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
        })

        # Test set evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # Save metrics for DVC
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
        cm_df.to_csv("metrics/confusion_matrix.csv")
        mlflow.log_artifact("metrics/confusion_matrix.csv")

        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
        )

        # Save run ID for orchestrator
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/last_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"\nTraining complete!")
        print(f"  Run ID:   {run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return run.info.run_id


if __name__ == "__main__":
    train()
```

### Metrics Gate Script

```python
# scripts/check_metrics.py
import json
import argparse
import sys


def check_metrics(metrics_file: str, min_accuracy: float, min_auc: float) -> bool:
    """
    Read metrics from file and assert they meet minimum thresholds.
    Exits with code 1 if any metric fails — used as a CI gate.
    """
    with open(metrics_file) as f:
        metrics = json.load(f)

    print(f"Checking metrics from {metrics_file}:")
    print(json.dumps(metrics, indent=2))

    passed = True

    if metrics.get("accuracy", 0) < min_accuracy:
        print(f"FAIL: accuracy={metrics.get('accuracy', 'missing'):.4f} < min={min_accuracy}")
        passed = False
    else:
        print(f"PASS: accuracy={metrics.get('accuracy'):.4f} >= min={min_accuracy}")

    if metrics.get("roc_auc", 0) < min_auc:
        print(f"FAIL: roc_auc={metrics.get('roc_auc', 'missing'):.4f} < min={min_auc}")
        passed = False
    else:
        print(f"PASS: roc_auc={metrics.get('roc_auc'):.4f} >= min={min_auc}")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.92)
    parser.add_argument("--min-auc", type=float, default=0.97)
    args = parser.parse_args()

    if not check_metrics(args.metrics_file, args.min_accuracy, args.min_auc):
        print("\nMetrics check FAILED — deployment blocked")
        sys.exit(1)

    print("\nAll metrics checks PASSED — proceeding with deployment")
```

### API Integration Test

```python
# tests/test_api.py — integration tests for the serving endpoint
import pytest
import httpx


BASE_URL = "http://localhost:8000"

SAMPLE_INPUT = {
    "mean_radius": 14.0,
    "mean_texture": 20.5,
    "mean_perimeter": 90.3,
    "mean_area": 600.0,
    "mean_smoothness": 0.1,
    "mean_compactness": 0.13,
    "mean_concavity": 0.09,
    "mean_concave_points": 0.05,
    "mean_symmetry": 0.18,
    "mean_fractal_dimension": 0.063,
}


@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=BASE_URL, timeout=10.0)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in [0, 1]

    assert "probability_benign" in data
    assert 0.0 <= data["probability_benign"] <= 1.0

    assert "probability_malignant" in data
    assert abs(data["probability_benign"] + data["probability_malignant"] - 1.0) < 1e-6


def test_predict_invalid_input(client):
    """Invalid input should return 422 Unprocessable Entity."""
    invalid_input = {**SAMPLE_INPUT, "mean_radius": -1.0}  # negative value
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422


def test_predict_missing_field(client):
    """Missing required field should return 422."""
    incomplete = {k: v for k, v in SAMPLE_INPUT.items() if k != "mean_radius"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_batch_predict(client):
    """Batch endpoint should handle multiple instances."""
    batch_input = {"instances": [SAMPLE_INPUT] * 10}
    response = client.post("/predict/batch", json=batch_input)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 10
    assert data["n_instances"] == 10


def test_response_time(client):
    """Single prediction should complete in under 200ms."""
    import time
    start = time.perf_counter()
    client.post("/predict", json=SAMPLE_INPUT)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, f"Response too slow: {elapsed_ms:.1f}ms"
```

### Prometheus Monitoring Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ml_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: "ml-api"
    static_configs:
      - targets: ["ml-api:8000"]
    metrics_path: /metrics

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

```yaml
# monitoring/ml_alerts.yml
groups:
  - name: ml_model_alerts
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.99, ml_request_latency_seconds_bucket) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ML API p99 latency exceeded 200ms"

      - alert: HighErrorRate
        expr: rate(ml_request_total{status="error"}[5m]) / rate(ml_request_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "ML API error rate exceeded 1%"

      - alert: ModelNotLoaded
        expr: up{job="ml-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML API is down"
```

### Complete Startup Script

```bash
#!/bin/bash
# scripts/start_local_stack.sh
# Start the full local MLOps stack

set -e

echo "=== Starting Local MLOps Stack ==="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker not found"; exit 1; }
command -v dvc >/dev/null 2>&1 || { echo "DVC not found — pip install dvc"; exit 1; }

# Pull data from DVC remote (if configured)
echo "Pulling data..."
dvc pull data/ || echo "DVC pull skipped (no remote configured)"

# Start infrastructure
echo "Starting Docker Compose stack..."
docker-compose up -d postgres mlflow prometheus grafana

# Wait for MLflow to be ready
echo "Waiting for MLflow..."
until curl -sf http://localhost:5000/health > /dev/null; do
    sleep 2
done
echo "MLflow is ready"

# Run training pipeline
echo "Running training pipeline..."
python -m src.train

# Check metrics gate
echo "Checking metrics..."
python scripts/check_metrics.py \
    --metrics-file metrics/eval_metrics.json \
    --min-accuracy 0.92 \
    --min-auc 0.97

# Register model
echo "Registering model..."
python scripts/register_model.py

# Start serving API
echo "Starting ML API..."
docker-compose up -d ml-api

echo ""
echo "=== Stack is running ==="
echo "  ML API:     http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  MLflow UI:  http://localhost:5000"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
```

---

## Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| **MLOps** | Practice of applying DevOps principles (CI/CD, version control, monitoring) to ML systems |
| **Experiment tracking** | Recording hyperparameters, metrics, and artifacts for every training run |
| **Model registry** | Centralized catalog of model versions with lifecycle stages |
| **Data drift** | Change in the statistical distribution of input features over time |
| **Concept drift** | Change in the relationship between features and target variable over time |
| **PSI (Population Stability Index)** | Metric quantifying the shift between two distributions; PSI > 0.2 signals significant drift |
| **KS test** | Two-sample Kolmogorov-Smirnov test; statistical test for whether two samples come from the same distribution |
| **DVC** | Data Version Control; tool for versioning large data files alongside code in git |
| **ONNX** | Open Neural Network Exchange; framework-agnostic model serialization format |
| **ONNX Runtime** | Inference engine for ONNX models; typically 2-5x faster than native frameworks |
| **Feature store** | Centralized repository for computing and serving features consistently across training and serving |
| **Shadow deployment** | Running a new model in parallel with production, logging its predictions without serving them |
| **Canary deployment** | Gradually routing a small percentage of traffic to a new model |
| **Deployment gate** | Automated check (accuracy threshold, latency requirement) that must pass before a model is promoted |
| **CI/CD for ML** | Automated pipeline: code push → test → train → evaluate → deploy |
| **MLflow Tracking** | Component of MLflow that records and queries experiments |
| **W&B Sweeps** | W&B feature for Bayesian hyperparameter optimization |
| **Staged deployment** | Moving a model through Staging → Production with validation at each stage |
| **Retraining trigger** | Condition that initiates a new model training cycle (schedule, drift, performance) |
| **Prometheus** | Open-source time-series database and monitoring system |
| **Grafana** | Open-source dashboard and visualization tool for metrics |

---

## What's Next

In **Part 19 — The Capstone: End-to-End ML System**, we bring everything from the entire series together into a single, production-ready ML system. You will:

- Design a complete ML system from requirements to deployment
- Apply all techniques from Parts 1-18 in a unified project
- Implement a real-world dataset with messy data, class imbalance, and distributional challenges
- Build a training pipeline with DVC + MLflow that trains multiple model architectures and selects the best
- Deploy a FastAPI serving endpoint behind a load balancer
- Set up full monitoring with drift detection and automated retraining
- Write a CI/CD pipeline that automatically tests, trains, and deploys on every commit

The capstone is where the theory becomes a portfolio project. Every piece we have built — from gradient descent in Part 1 to the MLOps infrastructure in this article — will be assembled into something you can run, show, and be proud of.

---

*Part 18 of 19 — Machine Learning: A Developer's Deep Dive from Fundamentals to Production*

*Previous: [Part 17 — ML System Design: Architecture, A/B Testing, and Feature Stores](./ml-deep-dive-part-17.md)*
*Next: [Part 19 — The Capstone: End-to-End ML System](./ml-deep-dive-part-19.md)*
