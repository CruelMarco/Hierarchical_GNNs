# Hierarchical Graph Neural Networks

**Mohammad Shaique Solanki**  


---

## 📄 Overview

This repository hosts the slides and supporting material for a seminar on **Hierarchical Graph Neural Networks (GNNs)**. We explore why hierarchy matters in graphs, survey existing GNN approaches, and dive into the **DiffPool** mechanism for learning multiscale structure.


## 🎯 Introduction

- **Motivation**: Graph-structured data appear everywhere—from social networks to molecular structures.  
- **Objective**: Learn and exploit hierarchical (multiscale) features in graphs for better performance and interpretability.

---

## 🔍 Background

- **Graph Neural Networks (GNNs)**  
  - Message-passing paradigm: node features are transformed and aggregated uniformly across layers.  
  - Flat architectures often suffer from oversmoothing, scalability bottlenecks, and limited ability to capture multiscale patterns.

---

## 📊 Observing Hierarchy in Graph Datasets

- Standard GNNs (e.g., 3-layer GCN + global mean pooling) on benchmark datasets frequently:
  - Over-smooth node representations  
  - Scale poorly on large or variable-size graphs  
  - Fail to capture long-range, multilevel dependencies  
- Empirical test accuracy: **0.7232** (flat) vs. **0.78** using DiffPool.

---

## ⚙️ Fundamentals of Hierarchical GNNs

- Introduce a **coarsening matrix** $C^{(l)}$ to merge nodes into clusters:
  
  $$H^{(l+1)} = \sigma\bigl(\mathrm{AGG}\bigl(C^{(l)}\,H^{(l)}\bigr)W^{(l)}\bigr)$$

- Captures multiple levels of abstraction by alternating convolution and pooling.

---

## 🔄 The DiffPool Mechanism

- Learns soft assignments of nodes to clusters at each layer:

  - **Input**: adjacency $A$, features $H^{(l)}$  
  - **Learn**: assignment matrix $S^{(l)}\in\mathbb{R}^{n_l\times n_{l+1}}$  
  - **Output**: pooled graph $(A', H')$
- Demonstrated to improve accuracy and reduce oversmoothing.

---

## 🚧 Challenges & Limitations

- **Computational cost** of learned pooling on very large graphs  
- Need for **interpretable** cluster assignments  
- Balancing **structure preservation** vs. **model complexity**

---

## 🗂 Files

- Folder **code** contains the .ipynb file of the talk-torial 
- Folder **papers** contain the reference material on HGNN


---

## 📚 References


1. DiffPool: Differentiable Pooling for GNNs (Yun et al., 2019) 
   https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf
2. Chris Morris’s Benchmark Datasets for Graph Classification  
   https://chrsmrrs.github.io/datasets/docs/datasets/  
3. Springer article on graph coarsening  
   https://link.springer.com/article/10.1007/s40324-021-00282-x  
4. Diffpool Implementation
   https://github.com/d-stoll/diffpool?tab=readme-ov-file

---

## ❓ Questions

Feel free to open an issue or contact **Mohammad Shaique Solanki** at moso00002@stud.uni-saarland.de.

---

