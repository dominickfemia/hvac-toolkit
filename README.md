# HVAC Engineering Toolkit (Excel + Machine Learning)

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
```
y_pred = model.predict(X_test)
```

We then measure error using metrics like Mean Absolute Error (MAE) or Root Mean Square Error (RMSE). For interpretability, we might also calculate the max error or average percentage error relative to true friction factor values. This gives an idea of whether the model is within the ±5% range for most points, for example. The results (printed to console) will tell us if the model is sufficiently accurate. If not, we could iterate on hyperparameters or data (this process would be documented in comments so readers understand the tuning process).

6. **Generating the Lookup Table:** One practical outcome of this project is to produce a lookup table of friction factors that engineers could use directly (for example, in Excel or other tools) without needing the ML code. To do this, we use the trained XGBoost model to predict friction factors on a grid of Re and roughness values. The script programmatically creates a fine grid, for example:
```
import numpy as np
Re_range = np.logspace(3.7, 8, num=100)        # e.g., 5e3 to 1e8 on log scale
rr_range = np.logspace(-6, -2, num=50)         # e.g., 1e-6 to 1e-2 on log scale
grid = [(Re, rr) for Re in Re_range for rr in rr_range]
X_grid = pd.DataFrame(grid, columns=['Re', 'rel_roughness'])
# If training used log10 features, remember to transform X_grid accordingly
f_pred = model.predict(X_grid)
```

This produces predicted friction factors for each combination on the grid. We then reshape or organize this output into a table or matrix form. One approach is to create a 2D table with Re values as rows and ε/D values as columns (similar to a Moody chart in numeric form). The code will then save this table to a CSV file, for example friction_factor_lookup.csv. Extensive comments in the code explain how the table is structured (e.g., the order of rows and columns) so that a user can easily import it elsewhere. Note: This lookup table essentially mirrors the Moody diagram: one can interpolate within it to get friction factors without solving equations.

7. **Results and Sample Usage:** Finally, the implementation script may demonstrate a quick example of using the model or lookup table. For instance, given a specific Re and roughness, it can show the friction factor from the model vs. what Colebrook’s equation would give (for validation). This helps confirm that the ML model is reasonable. We cite an example in the README (e.g., “For Re = 1e5 and ε/D = 0.001, the model predicts f ≈ 0.02, which is consistent with classical results.”). Such comparisons reassure the reader that the ML approach is producing physically plausible outputs.

Throughout the code, inline comments and markdown (if using a notebook) provide explanations for each step, making the process transparent. Even someone not deeply familiar with XGBoost or Python should be able to follow the logic from data input to final output.

## Understanding XGBoost’s Internal Workings (Educational Demo)

To bridge the gap between just using an ML library and understanding how it works, the repository includes a second code module (in a separate folder) that delves into the inner mechanism of gradient boosting. This section is an aside for the curious reader or those interested in ML; it recreates a simplified version of what XGBoost does when training the model. By studying this, one can better appreciate how the friction factor model is built “under the hood.” Key elements of this demo include:

- **Gradient Boosting Concept:** We start with a brief explanation that gradient boosting builds models in stages. The idea is to start with an initial prediction (e.g., the average friction factor in the training set) and then iteratively add decision trees that predict the residual errors left by the previous model. The code comments outline this concept before diving into implementation.

- **Simplified Implementation:** Rather than writing a full tree-building algorithm from scratch (which is quite complex), we leverage existing tools in a didactic way. For example, we might use sklearn.tree.DecisionTreeRegressor as the base learner to illustrate boosting. The demo code could do something like:
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

This loop mimics the boosting process: each tree is trying to predict the residual (error) from the previous approximation. We use only a few boosting rounds in this demonstration for clarity. After this, model_trees would contain a sequence of trees that together form an ensemble.

- **Applying the Ensemble:** The code then shows how to use the ensemble to predict new values. Essentially, one would sum the predictions of all the trees along with the initial prediction. For a given input $(\text{Re}, \epsilon/D)$:
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

The above function accumulates contributions from each weak learner. Comments in the code make it clear that this is essentially what XGBoost does, with additional enhancements like learning rate (shrinkage) and advanced split criteria. We might even incorporate a learning rate in the loop (e.g., multiply each tree’s prediction by a factor like 0.1) to show how it affects convergence.

- **Discussion:** After running the simplified booster, the script prints out a comparison between the ensemble’s predictions and the true values for a few examples. This confirms that even a small number of trees can start to capture the relationship. The code is thoroughly commented to point out how the residuals decrease with each round, illustrating the “gradient descent” aspect of gradient boosting. There may also be remarks on differences between this simplistic approach and XGBoost (for example, XGBoost uses a more sophisticated objective with second-order gradients, regularization terms, and can build trees in parallel at each step. These are mentioned conceptually but not implemented in our simple code).

By stepping through this process, a reader can conceptually connect the dots between the friction factor model we trained and the boosting algorithm that produced it. It demystifies the ML model: rather than a “black box,” it is shown as a sum of decision rules that approximate the Colebrook equation behavior. For those interested, references to XGBoost documentation or resources are provided to learn more about the full algorithm beyond this overview.

## Repository Structure and Usage

To keep things organized, the repository is structured into clear sections. All content is contained within this single GitHub project (no external submodules), using folders to separate the main components. Below is the layout of the repository and guidance on how to navigate it:

- README.md: (You are reading it!) The README serves as a summary and guidance document. It introduces the project, explains the methodology, and points to the relevant code and data. This makes the repo understandable as a standalone resource for someone who hasn’t seen the Notion page.

- /data Folder: Contains the dataset files. Notably, digitized_friction_data.csv resides here, which includes the Re, relative roughness, and friction factor data points collected from the Moody chart and Nikuradse experiments. Having a dedicated data folder keeps the repository tidy and allows easy updates or addition of data files (for example, if we add more points or a different dataset in the future).

- /model_training Folder: Contains the main machine learning implementation. For instance, this folder might have a Jupyter Notebook (XGBoost_friction_factor.ipynb) or a Python script (train_xgboost_model.py). This code takes you through importing the data, training the XGBoost model, evaluating it, and producing the lookup table CSV. If using a notebook, it may also include visualizations (e.g., plotting the learned friction factor curve against actual data points for a sample roughness) to verify the model’s fit. The folder could also include the output file friction_factor_lookup.csv (the table of predicted friction factors) if we choose to store it in the repo for reference. Every piece of code in this folder is heavily commented as described earlier.

- /xgboost_internal_demo Folder: Contains the educational demo code for XGBoost internals. For example, gradient_boosting_demo.py or a notebook by the same name, which implements the step-by-step gradient boosting procedure. This folder might also include a short README or notes explaining its purpose, but the code itself is written to be self-explanatory with comments. We isolate this in its own folder to avoid confusing users who are only interested in running the main model – it’s an optional deep-dive.

*(It’s worth noting that using two separate folders for the two code portions is purely for clarity; they belong to the same project/repository. On GitHub, a repository can certainly contain multiple folders – there is no need for separate projects. The README links to each section so users can easily find what they need.)*

- Environment and Dependencies: Although not a folder, we include info on how to set up and run the code. In the README or a requirements.txt file, we list dependencies like XGBoost, pandas, numpy, scikit-learn, etc. A user with Python 3.x can install these and reproduce the results. If there are any special instructions (e.g., needing to use Jupyter to view the notebooks), we mention them here. However, the code is straightforward and should run on any standard Python environment after installing the libraries.

**Using the Toolkit:** To use the trained model or lookup table, one can either run the notebook/script to regenerate everything or directly utilize the provided lookup CSV. For example, if an engineer wants a quick friction factor for a given Re and ε/D, they can open the CSV and interpolate between nearest values (much like reading a Moody chart). Alternatively, they can integrate the provided model (we could even save the trained XGBoost model to a file using Python’s joblib or pickle, but that might not be necessary for transparency). Instructions for these use-cases are provided in the repository documentation.

Finally, the repository includes references to the original Notion page or report for those who want a more narrative explanation of the project background. By keeping the GitHub content focused and technical, and the portfolio content more conceptual, we ensure each platform adds value in its own way.

## Conclusion

This GitHub repository is the technical companion to the friction factor toolkit, showcasing how modern machine learning can solve a classic engineering problem in fluid mechanics. Readers can expect to come away with not only a working model for friction factor prediction but also an understanding of how and why it works. By organizing the content as described and providing clear documentation, the project stands alone as a reproducible and educational resource. We encourage users to explore the code, tweak the model, or even contribute improvements. Whether one’s interest is in fluid dynamics or in machine learning (or both), this toolkit provides a concrete example of their intersection – using data and algorithms to augment traditional engineering methods in a practical, interpretable way.
