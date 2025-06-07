# 🌫️ DMGENet

**Dynamic Multi-Graph Ensemble Neural Network for Short-Term Air Quality Forecasting**

---

## 📝 Abstract

This paper proposes **DMGENet**, a novel deep learning framework for **short-term air quality forecasting**. To better capture complex spatiotemporal dependencies, the model integrates multi-graph spatial structures, advanced temporal modeling, and a dynamic ensemble mechanism based on reinforcement learning. Experimental results demonstrate that DMGENet outperforms existing baselines in prediction accuracy, showing strong adaptability and effectiveness in real-world air quality scenarios.

---

## 📊 Dataset

The air quality data is collected from **12 observing stations** around **Beijing**, covering the years **2013–2017**.  
It includes hourly measurements of various air pollutants and meteorological variables.  
The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi+Site+Air+Quality+Data).

---

## 🚀 Key Contributions

1. **Multi-Graph Construction**  
   A novel construction method encodes diverse inter-station spatial relationships, providing the model with richer prior structural information.

2. **Spatiotemporal Feature Extraction Module**  
   A hybrid module that combines a **Gated Temporal Convolutional Network (Gated TCN)** and a **Hybrid Graph Learning Module (HGLM)** to separately capture temporal and spatial dependencies, enhanced by an **Adaptive Spatiotemporal Attention Mechanism (ASTAM)** to address spatiotemporal heterogeneity.

3. **Dynamic Ensemble via Reinforcement Learning**  
   A dynamic ensemble strategy using an improved **Deep Deterministic Policy Gradient (DDPG)** algorithm to adaptively fuse outputs from multiple graph-based models.

---

## 🧩 Model Architecture

### 🔷 Overall Framework of DMGENet

![Model Architecture](Figures/Fig.1.png)  
*The proposed DMGENet mainly consists of two components: the Multi-Graph Prediction Module and the Ensemble Prediction Module.*

---

### 🔶 Adaptive Spatiotemporal Attention Mechanism (ASTAM)

<p align="center">
  <img src="Figures/Fig.2.png" alt="ASTAM" width="600"/>
</p>  
*Captures spatiotemporal heterogeneity by assigning dynamic attention weights across spatial and temporal dimensions.*

---

### 🧠 DDPG Actor Network in RLMC

![DDPG Actor](Figures/Fig.3.png)  
*Learns to generate dynamic fusion weights for multi-graph outputs during ensemble prediction.*

---

### 🛠️ Training Pipeline of RLMC

![RLMC Training](Figures/Fig.4.png)  
*Illustrates the training process of the Reinforcement Learning-based Multi-Graph Ensemble Method using improved DDPG.*

---

## ⚙️ Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/DMGENet.git
   cd DMGENet
