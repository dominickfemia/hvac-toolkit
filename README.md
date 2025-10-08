# HVAC Engineering Toolkit (Excel + Machine Learning)

*This is the technical companion to my [toolkit portfolio.](https://gomechra.com)*

## Introduction

This repository complements my project’s portfolio page by focusing on the **machine learning extension** of the provided friction calculators.
It provides enough context to stand on its own while remaining more concise than the full portfolio documentation.
The toolkit addresses the classic **Colebrook-White equation** for fluid-flow friction factor, using modern data-driven methods.

In traditional engineering practice, this equation is *implicit* and must be solved iteratively, an approach that becomes time-consuming for large or repeated calculations.
The well-known **Moody diagram**, compiled from thousands of experimental tests (including Nikuradse's pipe-flow experiments), has long been used to estimate friction factor graphically.
This project leverages **machine learning (ML)** to directly predict the **Darcy–Weisbach friction factor** directly from flow conditions, providing a fast, explicit, and highly accurate alternative to manual or iterative solutions.

## Project Overview and Scope

Within the **duct** and **pipe** modules, the toolkit predicts the **Darcy friction factor** from two key inputs: **Reynolds number (Re)** and **relative roughness (ε/D)**, the same parameters used in the Colebrook-White equation and Moody chart.

This repository provides:

- A **concise recap of the problem** and underlying methodology.
- A clear explanation of **why machine learning is valuable** in this context.
- Details on **how the training data** (digitized from the Moody chart and Nikuradse's experiments) was prepared.
- **Fully commented Python code** implementing an XGBoost regression model to learn the friction factor relationship.
- **Supplementary example code** illustrating the internal mechanics of gradient boosting for educational purposes.

## Why Use Machine Learning for the Colebrook Equation?

Using **machine learning (ML)** for this problem offers two main advantages:
1. Eliminates the need for iterative solutions.
2. Anchors predictions to real experimental data.

The **Colebrook–White equation** is implicit (the friction factor *f* appears on both sides), so it must be solved using iterative numerical methods or empirical approximations. In practice, each evaluation can require multiple computational steps, which become cumbersome when repeated across large systems or optimization routines.
An ML model, once trained, provides an **explicit predictive function; given *Re* and *ε/D*, it outputs *f* directly, with no iteration required.

Additionally, this approach is inherently **data-driven**. Instead of relying on a single empirical formula, an ML model can be trained on a broad set of known (Re, ε/D, f) points, capturing the nonlinear relationships across laminar, transitional, and turbulent regimes.

In this project, the model was trained on **digitized Moody and Nikuradse chart data**, grounding its predictions in real observed behavior. Because the method is flexible, the model can easily be **retrained or extended** as new data becomes available.

## Data Collection and Preparation

The first step was to assemble a dataset of known friction factor values by digitizing classical experimental results from **Moody (1944)** and **Nikuradse (1933)** charts. Digitizing both sources ensured consistency and overlap (particularly around values corresponding to *ε/D = 0.001*) while covering a broad range of **Reynolds numbers** and **relative roughness**.

The Moody data provided smoother surface conditions, whereas the Nikuradse data contributed higher roughness curves. Together, they form a robust, complementary dataset.

**Range of represented values:**

| **Source**     | **Reynolds Number (Re)** | **Friction Factor (f)** | **Relative Roughness (ε/D)** |
|------------|-----------------------|---------------------|---------------------------|
| Nikuradse  | 550 – 1.2 × 10⁶       | 0.0189 – 0.143      | 0.0010 – 0.0333           |
| Moody      | 3.1 × 10³ – 9.9 × 10⁷ | 0.008 – 0.045       | 0.00001 – 0.0010          |

The two resulting datasets were compiled into a table (CSV file) with the following columns: Reynolds number, relative roughness (ε/D), and friction factor $f$. Note that the raw data is represented in this repo under the file name *digitized_data.csv* as a placeholder for code demonstration. The actual data set is integrated into the toolkit and not available as a stand-alone file.

## Model Selection: XGBoost Regression

I chose XGBoost (Extreme Gradient Boosting) as the regression algorithm for this model due to its proven performance on tabular datasets. XGBoost is an open-source library that implements a powerful form of gradient-boosted decision trees. In gradient boosting, an ensemble of shallow “weak learner” trees are built sequentially, each new tree correcting the errors of the previous ones, which effectively minimizes the overall prediction error. XGBoost’s implementation is known for being fast and highly optimized, featuring parallelized tree building and advanced regularization to prevent overfitting.

For friction factor prediction, using XGBoost offers a few benefits:
- Nonlinear function approximation: The relationship between $f$ and $(\text{Re}, \epsilon/D)$ is highly nonlinear as evidenced by the curves on the Moody chart. XGBoost can model complex nonlinear relationships by combining many decision tree splits, capturing the curvature in the data.
- Speed and efficiency: Despite thousands of data points, XGBoost can train the model quickly. Once trained, predictions are almost instantaneous, which is great for engineering tools with repetitive calculations.
- Accuracy and tuning: XGBoost provides hyperparameters (tree depth, number of trees, learning rate, etc.) that can be tuned to improve accuracy. In practice, even default settings produce a reasonable model, but refinement options allow for greater overall accuracy.

Simply put, XGBoost served as my regression engine to learn $f = F(\text{Re}, \epsilon/D)$ from the data.

## XGBoost Model Implementation (Code Outline)

This repository contains a fully commented Python script that walks through the entire model development process, from reading data to outputting results. This script is designed to be easy to follow, even for those new to machine learning. Below is a step-by-step guide:

1. **Importing Libraries:** Use Python’s scientific stack – primarily pandas for data handling, numpy for numeric operations, and xgboost (along with scikit-learn wrappers) for the regression model. All required libraries are listed at the top of the script.
```
code
```

2. **Loading the Dataset:** Read the CSV file of digitized points into a pandas DataFrame. Inspect the data (printing out a few samples) to ensure it loaded correctly:
```
import pandas as pd
data = pd.read_csv('data/digitized_friction_data.csv')
print(data.head())
```

The DataFrame contains columns representing Reynolds number, relative roughness, and friction factor. Confirm that these match expectations (e.g., Re ranges and values make sense).

3. **Feature Engineering:** Pprepare the input features and target variable:
```
X = data[['Re', 'rel_roughness']].copy()
y = data['friction_factor'].copy()
```

Split the data into a training set and a test set, using a common ratio (80/20 here). Use sklearn.model_selection.train_test_split for reproducibility. The split ensures the model is tested on data it doesn't see during training.

4. **Training the XGBoost Regressor:** Initialize an XGBoost regressor (for example, using XGBRegressor from xgboost.sklearn module) with some default or tuned hyperparameters:
```
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)
```

The following explains each parameter: n_estimators is the number of trees, max_depth controls how complex each tree can be, and learning_rate scales the contribution of each new tree (preventing overfitting by making training more gradual).

5. **Model Evaluation:** After training, check how well the model learned the data. The code computes predictions on the held-out test set:
```
y_pred = model.predict(X_test)
```

Measure error using metrics like Mean Absolute Error (MAE) or Root Mean Square Error (RMSE). This gives an idea of whether the model is within the ±5% range for most points, for example. The results (printed to console) will demonstrate if the model is sufficiently accurate. If not, iterate on hyperparameters or data.

6. **Generating the Lookup Table:** Produce a lookup table of friction factors that can be used directly without needing the machine learning code. Use the trained XGBoost model to predict friction factors on a grid of Re and roughness values. The script programmatically creates a fine grid, for example:
```
import numpy as np
Re_range = np.logspace(3.7, 8, num=100)        # e.g., 5e3 to 1e8 on log scale
rr_range = np.logspace(-6, -2, num=50)         # e.g., 1e-6 to 1e-2 on log scale
grid = [(Re, rr) for Re in Re_range for rr in rr_range]
X_grid = pd.DataFrame(grid, columns=['Re', 'rel_roughness'])
# If training used log10 features, remember to transform X_grid accordingly
f_pred = model.predict(X_grid)
```

Reshape or organize the output into a table or matrix form. Save this table to a CSV file.

7. **Results and Sample Usage:** Across non-error producing input bounds (realistic observable data), the machine learning model produced results consistent with the Churchill approximation witin 0.2 to 4.9%.

## Understanding XGBoost’s Internal Workings

To bridge the gap between just using a machine learning library and understanding how it works, this repository includes a second code breakdown that delves into the inner mechanism of gradient boosting. This section recreates a simplified version of what XGBoost does when training a model.

- **Gradient Boosting Concept:** The idea is to start with an initial prediction (e.g., the average friction factor in the training set) and then iteratively add decision trees that predict the residual errors left by the previous model.

- **Simplified Implementation:** Rather than writing a full tree-building algorithm from scratch (which is quite complex), the example below leverages existing tools in a didactic way:
```
from sklearn.tree import DecisionTreeRegressor

# Start with an initial prediction (e.g., mean of y)
initial_pred = y_train.mean()
y_pred_current = np.full(y_train.shape, initial_pred)
model_trees = []  # to store the sequence of trees
n_boost_rounds = 3  # for demonstration, build 3 trees sequentially

for i in range(n_boost_rounds):
    # Compute residuals (current errors)
    residuals = y_train - y_pred_current
    # Train a small decision tree on residuals
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X_train, residuals)
    model_trees.append(tree)
    # Update the current prediction by adding the new tree’s predictions
    y_pred_current += tree.predict(X_train)
```

This loop mimics the boosting process: each tree is trying to predict the residual (error) from the previous approximation. Only a few boosting rounds are used in this demonstration for clarity. After this, model_trees would contain a sequence of trees that together form an ensemble.

- **Applying the Ensemble:** Use the ensemble to predict new values. Essentially, sum the predictions of all the trees along with the initial prediction. For a given input $(\text{Re}, \epsilon/D)$:
```
def predict_ensemble(Re, rr):
    # Start with initial prediction
    pred = initial_pred
    for tree in model_trees:
        pred += tree.predict([[Re, rr]])[0]
    return pred

sample = X_test.iloc[0]
print("Ensemble prediction:", predict_ensemble(sample['Re'], sample['rel_roughness']))
print("Actual friction factor:", y_test.iloc[0])
```

The above function accumulates contributions from each weak learner. After running the simplified booster, the script prints out a comparison between the ensemble’s predictions and the true values for a few examples. This confirms that even a small number of trees can start to capture the relationship.

**Using the Toolkit:** To use the trained model, directly utilize the generated lookup CSV table.

## 🤖 XGBoost Implementation

<details>
<summary>Click to expand Python code</summary>

```python
# train_xgboost.py
# Fully commented script: load data, train XGBoost, export lookup table

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
data = pd.read_csv("data/digitized_friction_data.csv")
X = data[['Re', 'rel_roughness']]
y = data['friction_factor']

# Optional: transform features (log scale helps with range)
X['logRe'] = np.log10(X['Re'])
X['logRR'] = np.log10(X['rel_roughness'])
X = X[['logRe', 'logRR']]

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train XGBoost
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# 5. Export lookup table
Re_range = np.logspace(3.7, 8, num=100)
rr_range = np.logspace(-6, -2, num=50)
grid = [(Re, rr) for Re in Re_range for rr in rr_range]
grid_df = pd.DataFrame(grid, columns=['Re','rel_roughness'])
grid_df['logRe'] = np.log10(grid_df['Re'])
grid_df['logRR'] = np.log10(grid_df['rel_roughness'])

grid_df['f_pred'] = model.predict(grid_df[['logRe','logRR']])
grid_df.to_csv("data/friction_factor_lookup.csv", index=False)
