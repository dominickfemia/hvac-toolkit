# HVAC Engineering Toolkit (Excel + ML)

This repository contains the **engineering backbone** behind my HVAC Toolkit:
- Code to train an **XGBoost** surrogate for Darcy friction factor and export a dense **(Re, ε/D) → f** lookup grid.
- Concise method/integration notes for using the grid in Excel.
- Tiny tests and sample data to verify the export.

> 📌 **Deliberate split:** This repo omits the Excel workbook and full visual/narrative.  
> For the complete case study (screenshots, worked examples, validation story), see **Portfolio:** https://gomechra.com

## What this repo includes
- `ml/` XGBoost training + grid export
- `docs/` terse method & Excel integration specs
- `data/` tiny sample points + (generated) grid
- `tests/` smoke tests for sanity

## What this repo intentionally omits
- The Excel workbook (`*.xlsm`) and high-res screenshots
- Large/raw datasets
