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

- **What it is:** Machine learning surrogate for Darcy–Weisbach friction factor.  
- **Why ML?** Avoids iterative Colebrook solves and anchors to real data (Moody + Nikuradse).  
- **What’s here:**  
  - `data/` → digitized dataset samples  
  - `model_training/` → full XGBoost pipeline (train → export lookup table)  
  - `xgboost_internal_demo/` → educational gradient boosting demo  
  - `docs/` → brief method notes  

## Quickstart
```bash
pip install -r requirements.txt
python model_training/train_xgboost.py
