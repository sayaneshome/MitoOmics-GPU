# MitoOmics-GPU [Work in Progress]

GPU-accelerated multi-omics pipeline to quantify and visualize the *Mitochondrial Health Index (MHI)* by integrating extracellular vesicle/mitochondrial-derived vesicle (EV/MDV) proteomics with single-cell RNA-seq.

Hackathon project by *Team Go Getters* at the NVIDIA Accelerate Omics Hackathon (8-25 Sept 2025).

## 👥 Team Go Getters

* *Sayane Shome, PhD* (AI in Healthcare, Stanford)[Team Lead]
* *Seema Parte, PhD* (Ophthalmology, Stanford)
* *Hirenkumar Patel, PhD* (Ophthalmology, Stanford)
* *Ankit Maisuriya* (PhD candidate, Quantum Photonics, Northeastern)
* *Medha Bhattacharya* (CS undergrad, UC Irvine)

---

## 🚀 Project Objective

* Develop a *GPU-accelerated pipeline* for mitochondrial health analysis.
* Link blood-derived EV/MDV proteomics with mitochondrial DNA copy-number proxies from scRNA-seq.
* Provide interpretable measures:

  * *Biogenesis* (capacity to grow new mitochondria)
  * *Fusion/Fission* (structural remodeling)
  * *Mitophagy* (repair/recycling)
  * *Heterogeneity* (variation across cells).
* Output: a unified *Mitochondrial Health Index (MHI)* summarizing mitochondrial resilience, fitness, and disease risk.

---

## 🖥️ GPU Acceleration

* Optimized with RAPIDS + GPU backends.
* Clear *CPU vs GPU speedups* for large datasets.
* Open-source, designed for integration with *scverse/rapids-singlecell*.


## 📊 Key Insights

* Unified mitochondrial health scoring (MHI).
* Patient-level and cell-type–level insights.
* Supports biomarker discovery, disease progression prediction, and drug response stratification.

---

## 🔮 Future Directions

* Add modalities: scATAC, metabolomics, spatial transcriptomics.
* Deploy web-server / pip package for biologist-friendly use.
* Clinical validation with partners & cohorts.
* ML upgrades for pattern discovery & prediction on MHI.


## 📬 Contact

📧 [sshome@stanford.edu](mailto:sshome@stanford.edu)
