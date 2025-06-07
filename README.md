# 🌫️ DMGENet

**Dynamic Multi-Graph Ensemble Neural Network for Short-Term Air Quality Forecasting**

---

## 📝 Abstract

Air quality forecasting plays a crucial role in urban environmental management and public health protection. However, existing methods still face challenges in modeling complex spatiotemporal relationships, integrating multi-source spatial information, and achieving dynamic adaptability. To address these issues, this paper proposes a Dynamic Multi-Graph Ensemble Neural Network (DMGENet) for short-term air quality forecasting. Specifically, multiple spatial graph structures are constructed from four perspectives—geographical distance, adjacency, semantic correlation, and functional similarity—to comprehensively encode spatial dependencies between monitoring stations. Subsequently, a novel spatiotemporal feature extraction framework is designed by integrating gated temporal convolution with a hybrid graph learning module, along with an adaptive spatiotemporal attention mechanism to effectively tackle spatiotemporal heterogeneity. Finally, a dynamic ensemble method based on an improved deep reinforcement learning strategy is introduced to adaptively combine the prediction results of multiple graph models, thereby enhancing model flexibility and forecasting accuracy. Experimental results show that, for the 6-hour ahead forecasting task, the proposed model achieves average reductions of 11.44% in RMSE and 12.50% in MAE compared to GC-LSTM, demonstrating its superior performance in short-term air quality forecasting.

---

## 📊 Dataset

The air quality data is collected from **12 observing stations** around **Beijing**, covering the years **2013–2017**.  
It includes hourly measurements of various air pollutants and meteorological variables.  
The dataset is publicly available from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).

---

## 🚀 Key Contributions

1. **Multi-Graph Construction**  
   We propose a multi-graph construction method that encodes diverse inter-station spatial relationships, thereby providing the model with richer prior structural information.

2. **Spatiotemporal Feature Extraction Module**  
   We develop a novel spatiotemporal feature extraction module that integrates a Gated Temporal Convolutional Network (Gated TCN) and a Hybrid Graph Learning Module (HGLM) to separately capture temporal and spatial dependencies, and further incorporates an Adaptive Spatiotemporal Attention Mechanism (ASTAM) to effectively address the inherent spatiotemporal heterogeneity in air quality prediction.

3. **Dynamic Ensemble via Reinforcement Learning**  
   We introduce a dynamic multi-graph ensemble method based on an improved Deep Deterministic Policy Gradient (DDPG) algorithm to adaptively integrate predictions from multiple graph-based models, thereby improving overall forecasting performance.

---

## 🧩 Model Architecture

### 🔷 Overall Framework of DMGENet

![Model Architecture](Figures/Fig.1.png)  
The proposed DMGENet mainly consists of two components: the Multi-Graph Prediction Module and the Ensemble Prediction Module.

---

### 🔶 Adaptive Spatiotemporal Attention Mechanism (ASTAM)

<p align="center">
  <img src="Figures/Fig.2.png" alt="ASTAM" width="600"/>
</p>  
Addresses spatiotemporal heterogeneity by assigning dynamic attention weights across spatial and temporal dimensions.

---

### 🧠 DDPG Actor Network in RLMC

![DDPG Actor](Figures/Fig.3.png)  
Learns to generate dynamic fusion weights for multi-graph outputs during ensemble prediction.

---

### 🛠️ Training Pipeline of RLMC

![RLMC Training](Figures/Fig.4.png)  
Illustrates the training process of the Reinforcement Learning-based Dynamic Multi-Graph Ensemble Method using improved DDPG.

---
