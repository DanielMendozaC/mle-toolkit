# MLflow Guide for ML Projects

## Overview

MLflow is an open-source platform designed to manage the complete machine learning lifecycle. This guide provides a general reference for integrating MLflow into your ML projects to track experiments, manage models, and ensure reproducibility.

## Installation

```bash
# Basic installation
pip install mlflow

# Full installation with extras
pip install mlflow[extras]
```

## Core Components

MLflow consists of four main components:

1. **MLflow Tracking**: Logs parameters, code versions, metrics, and artifacts when running ML code
2. **MLflow Projects**: Packages ML code in a reproducible format to share with other data scientists
3. **MLflow Models**: Provides a standard format for packaging ML models for a variety of deployment tools
4. **MLflow Model Registry**: Centrally manages models in your MLflow instance

## Project Configuration

### Basic Setup

Create a configuration file (e.g., `mlflow_config.py`) to maintain consistent settings:

```python
# mlflow_config.py
import mlflow
import os

# Set tracking URI
MLFLOW_TRACKING_URI = "file:./mlflow"  # Local filesystem
# Alternative: "sqlite:///mlflow.db"    # SQLite database
# Alternative: "postgresql://user:password@localhost:5432/mlflow"  # PostgreSQL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Default experiment name
DEFAULT_EXPERIMENT_NAME = "my-ml-project"

# Create experiment if it doesn't exist
def setup_experiment():
    experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            DEFAULT_EXPERIMENT_NAME,
            artifact_location=os.path.join("mlflow", DEFAULT_EXPERIMENT_NAME)
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
    return experiment_id
```

### Recommended Directory Structure

```
project-root/
├── mlflow/                     # MLflow data directory
│   ├── models/                 # Saved models
│   └── artifacts/              # Other output artifacts
├── mlflow_config.py            # MLflow configuration
├── start_mlflow_ui.py          # Script to launch the UI
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
├── data/                       # Dataset files
└── README.md                   # Project documentation
```

## Usage Examples

### Starting the MLflow UI

Create a script to easily launch the UI:

```python
# start_mlflow_ui.py
import mlflow
import subprocess
from mlflow_config import MLFLOW_TRACKING_URI, setup_experiment

if __name__ == "__main__":
    # Ensure experiment is set up
    setup_experiment()
    
    # Start MLflow UI
    subprocess.call(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])
```

Run with: `python start_mlflow_ui.py`

### Tracking Experiments

Basic usage pattern:

```python
import mlflow
from mlflow_config import setup_experiment

# Initialize experiment
experiment_id = setup_experiment()

# Start a run
with mlflow.start_run(run_name="experiment-name") as run:
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    
    # Your model training code here
    # ...
    
    # Log metrics during training
    for epoch in range(num_epochs):
        # Training logic
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
    
    # Log final metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log model
    # For scikit-learn:
    mlflow.sklearn.log_model(model, "model")
    # For PyTorch:
    # mlflow.pytorch.log_model(model, "model")
    # For TensorFlow/Keras:
    # mlflow.keras.log_model(model, "model")
    
    # Log additional artifacts
    mlflow.log_artifact("path/to/feature_importance.png")
    mlflow.log_artifact("path/to/confusion_matrix.png")
```

### Nested Runs

For complex workflows:

```python
with mlflow.start_run(run_name="parent-run") as parent_run:
    mlflow.log_param("dataset", "full_dataset")
    
    # Child run for data preprocessing
    with mlflow.start_run(run_name="data-preprocessing", nested=True):
        mlflow.log_param("scaler", "standard")
        # preprocessing code
    
    # Child run for feature selection
    with mlflow.start_run(run_name="feature-selection", nested=True):
        mlflow.log_param("feature_selector", "recursive")
        # feature selection code
    
    # Child run for model training
    with mlflow.start_run(run_name="model-training", nested=True):
        mlflow.log_param("model_type", "xgboost")
        # model training code
```

### Loading Models

```python
# Load a model from a specific run
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Load a registered model version
loaded_model = mlflow.sklearn.load_model("models:/my-model/Production")
```

## Model Registry Workflow

### Registering a Model

```python
# Register model from a run
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="my-model"
)

# Alternative approach with MLflow client
client = mlflow.tracking.MlflowClient()
client.create_registered_model("my-model")
```

### Managing Model Lifecycle

```python
client = mlflow.tracking.MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Production"
)

# Archive when no longer needed
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Archived"
)
```

## Artifact Tracking

### Saving Datasets

```python
# Save dataset as an artifact
import pandas as pd
import os

# Create directory for datasets
os.makedirs('save_data', exist_ok=True)

# Save training data
X_train.to_parquet('save_data/x_train.parquet')
mlflow.log_artifact('save_data/x_train.parquet')

# Save all files in directory
mlflow.log_artifacts('save_data/')
```

### Saving Visualizations

```python
import matplotlib.pyplot as plt
import os

# Create directory for images
os.makedirs('images', exist_ok=True)

# Create visualization
plt.figure(figsize=(10, 6))
# Plotting code...
plt.savefig('images/feature_importance.png')

# Log individual image
mlflow.log_artifact('images/feature_importance.png')

# Or log all images in directory
mlflow.log_artifacts('images')
```

## Using Autologging

```python
# Enable autologging for scikit-learn models
mlflow.sklearn.autolog()

# Train your model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Disable autologging if needed
mlflow.sklearn.autolog(disable=True)
```

## Best Practices

1. **Organize with experiments**: Create separate experiments for different project components or approaches

2. **Use descriptive run names**: Make run names informative about what changed
   ```python
   with mlflow.start_run(run_name="xgboost-hyperparameter-tuning-v2"):
   ```

3. **Tag runs for easier filtering**:
   ```python
   mlflow.set_tag("data_version", "v2.0")
   mlflow.set_tag("feature_set", "text_and_categorical")
   mlflow.set_tag("environment", "development")
   ```

4. **Log all relevant artifacts**:
   - Model visualizations
   - Performance curves (ROC, PRC)
   - Feature importance plots
   - Sample predictions
   - Data distribution plots

5. **Track environment dependencies**:
   ```python
   mlflow.log_artifact("requirements.txt")
   # or
   mlflow.sklearn.log_model(model, "model", conda_env=conda_env)
   ```

6. **Automate hyperparameter tuning**:
   ```python
   # Example with hyperopt
   from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
   
   def objective(params):
       with mlflow.start_run(nested=True):
           mlflow.log_params(params)
           # train model with params
           mlflow.log_metric("val_loss", val_loss)
           return {"loss": val_loss, "status": STATUS_OK}
   
   search_space = {
       "learning_rate": hp.loguniform("learning_rate", -5, 0),
       "max_depth": hp.choice("max_depth", range(1, 20))
   }
   
   with mlflow.start_run(run_name="hyperparameter-optimization"):
       best = fmin(objective, search_space, algo=tpe.suggest, max_evals=50, trials=Trials())
       mlflow.log_params(best)
   ```

## Advanced Configurations

### Using a Remote Tracking Server

```python
# Set up tracking URI to remote server
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Setting Up a Production Tracking Server

For team environments, set up a dedicated tracking server:

```bash
# Start MLflow server with PostgreSQL backend and S3 artifact store
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0
```

### Integration with Deep Learning Frameworks

For PyTorch:
```python
mlflow.pytorch.autolog()
```

For TensorFlow:
```python
mlflow.tensorflow.autolog()
```

## Common Issues & Troubleshooting

- **Issue**: Nothing is appearing in the MLFlow UI
  **Solution**: Make sure you start the UI from the directory where your notebook is running.
  
- **Issue**: I installed a library, but I'm getting an error loading it
  **Solution**: Be sure you installed it in the same environment as where your notebook is running.
  
- **Issue**: MLFlow doesn't seem to be working at all, I'm just getting errors all over the place
  **Solution**: This might be a versioning issue. Check that MLflow version is compatible with your Python environment.
  
- **Issue**: The UI broke, I can't get back to it
  **Solution**: Use this command to kill the process and restart it:
  ```bash
  sudo lsof -i :5000 | awk '{print $2}' | tail -n +2 | xargs kill
  ```

- **Issue**: Experiments not showing in UI
  **Solution**: Check tracking URI configuration and server connection

- **Issue**: Artifacts not being saved
  **Solution**: Verify artifact paths and permissions

- **Issue**: Inconsistent experiment data
  **Solution**: Always use `with mlflow.start_run()` context manager

- **Issue**: Large artifacts slowing down tracking
  **Solution**: Consider filtering or downsampling large files before logging

## Resources

- [Official MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models Documentation](https://mlflow.org/docs/latest/models.html)
- [MLflow Model Registry Documentation](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Projects Documentation](https://mlflow.org/docs/latest/projects.html)
