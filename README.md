# HVAC Engineering Toolkit

This repo is the **technical companion** to my [Notion portfolio](https://gomechra.com).

## Introduction

This GitHub repository complements the project’s Notion portfolio page by focusing on the machine learning extension of the friction factor toolkit. It provides a standalone overview of the project so readers can understand the context, while remaining more concise than the full Notion documentation. The toolkit addresses the classic Colebrook equation for pipe flow friction factor, using modern data-driven methods. In traditional engineering practice, the Colebrook–White equation is implicit and must be solved iteratively, which can be time-consuming for extensive calculations. The well-known Moody diagram was historically used to obtain friction factors graphically, based on a large compilation of experimental data (on the order of 10,000 tests, including Nikuradse’s pipe flow experiments). This project leverages machine learning (ML) to directly predict the Darcy–Weisbach friction factor from flow conditions, providing a fast, explicit solution without iteration. This GitHub repository serves as a technical deep-dive into that ML solution, showing how the model is built and how it relates to the underlying engineering problem.

## Project Overview and Scope

In essence, the toolkit predicts the pipe friction factor given two inputs: Reynolds number (Re) and relative roughness (ε/D), which are the same parameters used in the Colebrook equation and Moody chart. The Notion page covers background equations and basic methods (including manual/Excel-based solutions), whereas this repository extends the project with a machine learning approach. We focus on the technical story behind using ML for this problem, leaving out unrelated implementation details (for example, the earlier VBA/Excel calculations are omitted since they do not contribute to the ML aspect). The repository provides:

- A brief recap of the problem and methodology (so it’s clear this is the same project as in the Notion portfolio).
- An explanation of why machine learning is useful in this context.
- Details on how the training data was obtained from classic sources (the Moody chart and Nikuradse’s data).
- Fully commented code implementing an XGBoost regression model to learn the friction factor relationship.
- Additional commented code illustrating the internal workings of gradient boosting (to demystify what XGBoost does under the hood).

By including both the high-level implementation and a peek into the internal mechanics, the repository ensures that readers not only see the result but also gain some intuition for the ML techniques involved.

## Why Use Machine Learning for the Colebrook Equation?

Using ML for this problem offers two main advantages: avoiding iterative solutions and basing the model on real data. The Colebrook–White equation requires iterative numerical methods to find the friction factor because the formula is implicit (f appears on both sides). In practical terms, each evaluation of friction factor might require several computation steps or approximation techniques, which can be cumbersome especially when solving many times (e.g., in network simulations or optimization). A machine learning model, once trained, provides an explicit predictive function: given Re and roughness, it directly outputs the friction factor in one calculation, with no need for iteration.

Moreover, the ML approach is inherently data-driven. The Moody diagram itself was a data-driven solution compiled from experiments, with an estimated accuracy of about ±5–10%. Instead of relying on a specific empirical formula, we can train a model on a broad set of known $(\text{Re}, \epsilon/D, f)$ data points. If the dataset is representative, the ML model can capture the complex relationship across laminar, transitional, and turbulent regimes. In this project, we use published data points from Moody’s chart and Nikuradse’s experiments as the ground truth. By doing so, the model’s predictions are anchored to real observed behavior of friction factors, potentially improving accuracy and ensuring that subtle trends (like the shape of the transition zone curve) are learned. Using ML is also flexible – the model could be retrained or expanded if new data become available or if a different range of conditions needs to be covered.

Finally, exploring this approach has educational value. It demonstrates how artificial intelligence techniques can tackle classic engineering problems. Previous researchers have investigated applying machine learning to Moody chart data, finding that advanced regressors (e.g. Random Forests, support vector machines, etc.) can predict friction factor reasonably well. Our toolkit uses XGBoost, a state-of-the-art algorithm, to see if we can achieve a high-accuracy surrogate for the Colebrook equation.

## Data Collection and Preparation

The first step was to build a dataset of known friction factor values over a wide range of flow conditions. We obtained data by digitizing the Moody diagram and incorporating classic experimental results:

Moody Chart Data: We extracted points from the Moody diagram covering turbulent and transitional flow regimes for various relative roughness values. This involved reading values of friction factor $f$ at many combinations of Re and ε/D along the Moody curves. (In literature, researchers have done similar digitization; for example, one study sampled over 1,050 points from the Moody chart spanning $5×10^3 < \text{Re} < 10^8$ and $10^{-6} < \epsilon/D < 10^{-2}$.) Our data collection ensures that all relevant flow regimes are represented – from smooth pipes to rough pipes, and from the onset of turbulence through fully rough flow.

Nikuradse’s Data: To enrich the dataset, we also included data from Johannes Nikuradse’s classic experiments on sand-roughened pipes. Nikuradse’s results provide detailed friction factor measurements for pipes of known roughness, which underpin the theoretical roughness curves on the Moody diagram. Including these points helps ground the model in actual experimental observations beyond the graphical chart itself.

All told, the compiled dataset is a table (CSV file) of data points with columns: Reynolds number, relative roughness (ε/D), and the corresponding friction factor $f$. This CSV file (digitized_friction_data.csv) is included in the repository (under a /data folder) so that others can inspect it or reuse it. Brief method notes on how the data was digitized (e.g. tools or interpolation techniques used) can be found in the repository documentation, but the main point is that we have a sufficient and wide-ranging set of $(\text{Re}, \epsilon/D, f)$ samples to train our ML model.

Before training, we performed minimal preprocessing: since both Re and ε/D span several orders of magnitude, it’s common to apply a logarithmic scale to these features for easier learning. (For instance, the Moody chart itself is log-log scale in Re and f.) In our code, we can transform Re and possibly ε/D using log10 to help the model handle the nonlinearity. The friction factor output can remain in linear scale as it is bounded roughly between 0 and 0.1. We also shuffled and split the data into training and validation sets to be able to evaluate the model’s accuracy on unseen points.

## Model Selection: XGBoost Regression

We chose XGBoost (Extreme Gradient Boosting) as the regression algorithm for this task due to its proven performance on tabular datasets. XGBoost is an open-source library that implements a powerful form of gradient-boosted decision trees. In gradient boosting, an ensemble of shallow “weak learner” trees are built sequentially, each new tree correcting the errors of the previous ones, which effectively minimizes the overall prediction error. XGBoost’s implementation is known for being fast and highly optimized, featuring parallelized tree building and advanced regularization to prevent overfitting. In fact, XGBoost’s ability to provide parallel tree boosting and handle large datasets efficiently has made it a top choice in many machine learning competitions.

For our friction factor prediction, using XGBoost offers a few benefits:

- Nonlinear function approximation: The relationship between $f$ and $(\text{Re}, \epsilon/D)$ is highly nonlinear (as evidenced by the curves on the Moody chart). XGBoost can model complex nonlinear relationships by combining many decision tree splits, capturing the curvature and asymptotic behavior in the data.
- Speed and efficiency: Even though we have thousands of data points, XGBoost can train the model quickly. Once trained, predictions are almost instantaneous, which is great for engineering tools that might need to compute friction factors repeatedly.
- Accuracy and tuning: XGBoost provides hyperparameters (tree depth, number of trees, learning rate, etc.) that we can tune to improve accuracy. In practice, even default settings produce a reasonable model, but we have the option to refine it. Given that the original Moody chart had around 5–10% accuracy bounds, we aim to get the ML model’s predictions within a similar or smaller error margin compared to the true data.

In summary, XGBoost serves as our regression engine to learn $f = F(\text{Re}, \epsilon/D)$ from the data. By training this model, we essentially create an explicit mapping that mirrors the Moody chart: input roughness and Reynolds number, get friction factor out.

## XGBoost Model Implementation (Code Outline)

The repository contains a fully commented Python script (or Jupyter Notebook) that walks through the entire model development process, from reading data to outputting results. This script is designed to be easy to follow, even for those new to machine learning. Below is an outline of what the implementation covers:

1. **Importing Libraries:** We use Python’s scientific stack – primarily pandas for data handling, numpy for numeric operations, and xgboost (along with scikit-learn wrappers) for the regression model. All required libraries are listed at the top of the script. (If setting up the environment, the user should install XGBoost via pip or conda. The repository may include a requirements.txt noting the package versions for reproducibility.)

2. **Loading the Dataset:** The CSV file of digitized points is read into a pandas DataFrame. We then inspect the data (printing out a few samples) to ensure it loaded correctly. The code might look like:
```
import pandas as pd
data = pd.read_csv('data/digitized_friction_data.csv')
print(data.head())
```

The DataFrame contains columns like Re, rel_roughness, and friction_factor. We confirm that these match expectations (e.g., Re ranges and values make sense).

3. **Feature Engineering:** We prepare the input features and target variable. For example:
```
X = data[['Re', 'rel_roughness']].copy()
y = data['friction_factor'].copy()
```

Optionally, we apply a logarithmic transform to Re and rel_roughness here (this will be noted in the comments if done). We then split the data into a training set and a test (validation) set, using a certain ratio (commonly 80/20 or 70/30). This could use             sklearn.model_selection.train_test_split for reproducibility. The split ensures we can evaluate the model on data it hasn’t seen during training.

4. **Training the XGBoost Regressor:** We initialize an XGBoost regressor (for example, using XGBRegressor from xgboost.sklearn module) with some default or tuned hyperparameters. For instance:
```
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)
```

The code includes comments explaining each parameter: n_estimators is the number of trees, max_depth controls how complex each tree can be, learning_rate scales the contribution of each new tree (preventing overfitting by making training more gradual), etc. During training, XGBoost will output its progress (or we can suppress it) – the script might show training logs or simply state that training is complete.

5. **Model Evaluation:** After training, we check how well the model learned the data. The code will compute predictions on the held-out test set:
