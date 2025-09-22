# HVAC Engineering Toolkit (Excel + ML)

This repo is the **technical companion** to my [Notion portfolio](https://gomechra.com).

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

Background

The classic Colebrook equation is implicit and requires iteration . The Moody chart provides empirical f values but is ±5–10% accurate .
This repo shows how an XGBoost model can approximate those values directly, providing fast, explicit predictions.

(Full narrative, screenshots, and validation are on the Notion portfolio.)
