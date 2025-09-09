# HVAC Engineering Toolkit (Excel + ML)

A professional-grade Excel toolkit for HVAC and hydronic system design, built for engineers, designers, and interns. Includes calculators for duct, pipe, fan, and pump sizing, with an optional ML-based Darcy friction factor estimator trained on empirical data.

---

## 🚀 Features
- 4-in-1 calculator suite: duct (rectangular, round, flat oval), pipe, fan, pump
- Friction factor toggle: Churchill approximation or ML (XGBoost regression)
- Validation UI: green/red/gray states with clear messages
- Protected & versioned Excel workbook
- Fully documented (PDF User Guide, Instructions tab, Version Log)

---

## 🔧 How It Works
- **Inputs:** flow rate, geometry, run length, surface roughness, fluid properties
- **Friction factor estimation:** Churchill (explicit) or ML (trained on Moody/Nikuradse data, ~600 points)
- **Core calculations:** velocity, Reynolds number, Darcy–Weisbach, head/pressure loss
- **Outputs:** friction loss, horsepower, TDH, velocity
- **Validation:** range checks with color-coded messages

---

## 📊 Methods & Validation
- **Colebrook–White equation (reference)**
- **Churchill approximation (implemented)**
- **ML (XGBoost)** trained on digitized Moody/Nikuradse data
- Accuracy: R² ≈ 0.96, RMSE ≈ 0.0025

References:
- Nikuradse (1933), *Laws of Flow in Rough Pipes*
- Moody (1944), *Friction Factors for Pipe Flow*
- ASHRAE Handbook – Fundamentals (2021)
- Churchill (1977), *Friction-Factor Equation Spans All Fluid-Flow Regimes*

---

## 📸 Screenshots
*(Add images here, e.g. Duct calculator, ML toggle)*

---

## 🗺 Roadmap
- Automatic fitting loss library
- Expanded fluid property database (glycol %, refrigerants)
- Advanced input error handling
- Standalone app or web-based version

---

## 📬 Contact
info.help@gomechra.com  
For questions, feedback, or licensing inquiries.
