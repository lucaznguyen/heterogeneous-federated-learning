# Federated Learning: Exploring Statistical, Model, System, and Device Heterogeneity

**An interactive Python notebook implementing various heterogeneity challenges in Federated Learning, inspired by key papers in the FL literature.**

---

## 🔖 **Table of Contents**

1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Milestones & Techniques](#milestones--techniques)

   * [Milestone 0: Baseline FedAvg (Homogeneous)](#milestone-0-baseline-fedavg-homogeneous)
   * [Milestone 1: Statistical Heterogeneity (FedProx)](#milestone-1-statistical-heterogeneity-fedprox)
   * [Milestone 2: Model Heterogeneity (FedDF)](#milestone-2-model-heterogeneity-feddf)
   * [Milestone 3: System Heterogeneity (FedAsync)](#milestone-3-system-heterogeneity-fedasync)
   * [Milestone 4: Device Heterogeneity (Weighted FedAvg)](#milestone-4-device-heterogeneity-weighted-fedavg)
5. [Running the Notebook](#running-the-notebook)
6. [References & Citations](#references--citations)

---

## 📌 **Introduction**

Federated Learning (FL) enables distributed devices to collaboratively train a global model without sharing their raw data. However, FL faces multiple challenges, particularly related to various types of **heterogeneity**:

* **Statistical heterogeneity** (Non-IID data distributions).
* **Model heterogeneity** (clients having different model architectures).
* **System heterogeneity** (varying communication delays, asynchrony).
* **Device heterogeneity** (differences in computational resources).

This project systematically explores these heterogeneities, presenting clear implementations, visualizations, and comparisons of techniques addressing each challenge.

---

## ⚙️ **Environment Setup**

The notebook requires:

* **Python ≥ 3.8**
* **PyTorch**
* **torchvision**
* **tqdm**
* **Matplotlib**
* **Jupyter Notebook**

Install all dependencies via:

```bash
pip install torch torchvision tqdm matplotlib notebook
```

---

## 📂 **Project Structure**

The entire project is implemented in **one single notebook**, organized clearly into self-contained milestones. Each milestone introduces a new dimension of heterogeneity.

```
📔 Hetero-FL.ipynb
📄 README.md
```

---

## 🚩 **Milestones & Techniques**

### Milestone 0: Baseline FedAvg (Homogeneous)

* **Technique**: Federated Averaging (FedAvg) [\[McMahan et al. 2017\]](#references--citations)
* **Setting**: IID data, homogeneous clients
* **Formula** (FedAvg aggregation):

$$
w_{t+1} = \frac{1}{K}\sum_{k=1}^{K} w_{t+1}^{(k)}
$$

* **Visualization**: Training accuracy vs. elapsed time (steady improvement as baseline).

---

### Milestone 1: Statistical Heterogeneity (FedProx)

* **Technique**: FedProx [\[Li et al. 2020\]](#references--citations)
* **Purpose**: Regularize client updates under non-IID conditions
* **Formula** (FedProx objective):

$$
\min_{w} f_k(w) + \frac{\mu}{2}\|w - w_t\|^2
$$

* **Visualization**: Boxplot highlighting reduced accuracy variance across clients compared to standard FedAvg.

---

### Milestone 2: Model Heterogeneity (FedDF)

* **Technique**: Federated Distillation (FedDF) [\[Lin et al. 2020\]](#references--citations)
* **Purpose**: Aggregate heterogeneous local models through knowledge distillation
* **Formula** (FedDF distillation):

$$
L_{\text{KD}} = D_{\text{KL}}\left(\text{softmax}\left(\frac{z_{\text{teacher}}}{T}\right), \text{softmax}\left(\frac{z_{\text{student}}}{T}\right)\right)
$$

* **Visualization**: Bar chart showing FedDF significantly restores global accuracy despite heterogeneous models.

---

### Milestone 3: System Heterogeneity (FedAsync)

* **Technique**: FedAsync [\[Xie et al. 2019\]](#references--citations)
* **Purpose**: Handle asynchronous updates from heterogeneous client communication speeds.
* **Formula** (FedAsync weighted updates based on staleness):

$$
w_{t+1} = (1-\alpha)w_t + \alpha w^{(k)},\quad\alpha=\frac{\eta_0}{(s+1)^\gamma},\quad s=\text{staleness}
$$

* **Visualization**: Stacked bar chart showing the impact of deadline choices on the proportion of timely vs late updates.

---

### Milestone 4: Device Heterogeneity (Weighted FedAvg)

* **Technique**: Sample-weighted Federated Averaging
* **Purpose**: Fairly aggregate contributions from devices with different computational power (batch sizes, epochs).
* **Formula** (weighted FedAvg):

$$
w_{t+1} = \frac{1}{\sum_{k} n_k}\sum_{k=1}^{K} n_k w_{t+1}^{(k)},\quad n_k=\text{samples processed by client }k
$$

* **Visualization**: Bar chart displaying the total data contributions by device tiers, demonstrating fairness.

---

## 🚀 **Running the Notebook**

1. Clone this repository and navigate to the folder.
2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `Hetero-FL.ipynb` and select "Cell" → "Run All".

Each milestone can be run independently by collapsing/expanding cells in Jupyter.

---

## 📚 **References & Citations**

To cite original ideas and algorithms used, please refer to the following:

* **FedAvg**:
  McMahan et al. (2017).
  ["Communication-Efficient Learning of Deep Networks from Decentralized Data"](https://arxiv.org/abs/1602.05629).
  *AISTATS 2017*.

* **FedProx**:
  Li et al. (2020).
  ["Federated Optimization in Heterogeneous Networks"](https://arxiv.org/abs/1812.06127).
  *MLSys 2020*.

* **FedDF**:
  Lin et al. (2020).
  ["Ensemble Distillation for Robust Model Fusion in Federated Learning"](https://arxiv.org/abs/2006.07242).
  *NeurIPS 2020*.

* **FedAsync**:
  Xie et al. (2019).
  ["Asynchronous Federated Optimization"](https://arxiv.org/abs/1903.03934).
  *arXiv preprint*.

Additionally, for further exploration, the open-source implementation [FedAsync GitHub](https://github.com/yuxuan18/fedAsync) is recommended.

---

## 🎯 **Conclusion**

This notebook demonstrates the impact of different heterogeneity challenges in Federated Learning, providing detailed mathematical descriptions, clearly annotated code, and intuitive visualizations.

Each milestone illustrates important solutions from the literature, highlighting their effectiveness in handling real-world FL settings.

---

## 💡 **Next Steps & Suggestions**

* Experiment with other FL algorithms (e.g., FedDyn, FedNova, FedAvgM).
* Scale-up to larger datasets and more realistic simulations.
* Explore privacy enhancements (e.g., Differential Privacy).

---

⭐ **Happy Federated Learning!** 🌐✨
