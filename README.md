# ARC AGI Competition

This respiratory tracks my progress on working on the ARC competition. Solving this problem assists me in working with programming chain actions, decision making and workflow - areas where I find are useful in my project of building a humanoid.

## Folder Structure

- `/models`: Contains trained models.
- /output
  - /checkpoint: Checkpoint path to save and load models to resume training.
  - /results: Contains results from different model training sessions, including accuracy, loss plots, or model checkpoints.
- /utils: Utility scripts for tasks such as data preprocessing, metrics calculation, and visualization.
  - /analytics: Contains noted insights I have gathered while analysing the dataset.
- /experiments: Contains experimented models I have tried implementing from scratch or intuition - basically failed models. Each model has corresponding output folder.
- `/ARC-AGI`: Imported dataset from the competition's respiratory.

## Prerequisites

To run these experiments, you'll need to have Python installed along with the following libraries:

- `torch` (PyTorch)
- `tensorflow` (Optional, if you want to use TensorFlow models)
- `numpy`
- `pandas`
- `matplotlib` (for plotting results)
- `scikit-learn` (for data preprocessing)
