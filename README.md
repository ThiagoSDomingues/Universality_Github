🌌 Universality_Github
Companion Repository for:
“The shape of transverse momentum spectra in hybrid hydrodynamic models”

(The ExTrEMe Collaboration)

📘 Overview

This repository contains all materials associated with the analysis of the universal scaled transverse momentum spectra in hybrid hydrodynamic models, including:

* Bayesian prior design and training data

* Gaussian Process (GP) emulators for four viscous-correction prescriptions

* A fully interactive Streamlit application for exploring model parameter dependence

* Documentation and files necessary to reproduce the emulator-based predictions featured in the manuscript

The goal is to make the analysis transparent, reproducible, and easy to extend for future studies of universality, scaled spectra, and hydrodynamic modeling in heavy-ion collisions.

📂 Repository Structure
1. Bayesian_data/

Contains numerical inputs used for the Bayesian analysis, including:

* The event-average Universal scaled spectra for pions $U(x_T)$ for each design point; 
* Experimental data from ALICE collaboration Pb+Pb 2.76 TeV.

2. design_pts_Pb_Pb_2760_production/

Design points employed for the GP emulator training for Pb–Pb collisions at 2.76 TeV. Includes:

📌 Design point values

🏷️ Labels

🎯 Prior ranges for all calibrated parameters

These design points were produced by the JETSCAPE collaboration

3. emulators/

Pickle files containing the trained Gaussian Process emulators for the universal scaled spectra for all four viscous-correction prescriptions:

* emulators_Grad_6_pcs.pkl

* emulators_CE_6_pcs.pkl

* emulators_PTM_6_pcs.pkl

* emulators_PTB_6_pcs.pkl

Each file contains:

* GP models for the first six principal components of the scaled spectra

* Mean + variance predictions

4. streamlit_app_Bayesian.py 🚀

An interactive Streamlit dashboard that allows users to:

* Adjust hydrodynamic model parameters

* Instantly visualize their effect on the scaled spectra $U(x_T)$ for each viscous-correction choice (Grad, CE, PTM, PTB)

* Inspect emulator uncertainty through confidence bands

* Compare parameter sensitivity across the 4 δf models

5. Requirements Files

* requirements.txt – Python dependencies

* runtime.txt – Recommended runtime environment for Streamlit deployment

These ensure full reproducibility of the emulator predictions and web app environment.

🔧 Installation & Usage
git clone https://github.com/<youruser>/Universality_Github.git
cd Universality_Github
pip install -r requirements.txt

To launch the Streamlit exploration tool:
* streamlit run streamlit_app_Bayesian.py

🤝 Contributions

Contributions, suggestions, and issue reports are welcome. Feel free to open:

* Issues for bugs or feature requests;

* Pull requests for improvements;

* Discussions.

⭐ Acknowledgments

This work is part of the ExTrEMe Collaboration. 
The repository contains materials used in our Bayesian analysis pipeline, Gaussian Process emulation, and scaled-spectra universality investigation.
