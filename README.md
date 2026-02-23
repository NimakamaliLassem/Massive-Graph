# Graph Embeddings and GNN-Based Community Detection in Social and Computer Networks

**Authors:** Nima Kamali Lassem · Nicola Mambelli  
**Course:** Massive Graph — Polytechnique Montréal, February 2026

---

## Overview

This project studies **community detection** and **link prediction** on two contrasting real-world graph datasets. We progress from classical algorithms through spectral methods, shallow embeddings, and graph neural networks, all the way to knowledge graph embedding models and transfer learning — comparing every method on both datasets.

### Key Finding

> **No single model wins everywhere.** Dataset properties — symmetry, directionality, relation types, data density — determine which method performs best. Transfer learning from KGE to GCN achieves the best community detection NMI (0.680) on LastFM, surpassing all classical and spectral methods.

---

## Datasets

<table>
<tr><th></th><th>LastFM Asia</th><th>Cisco g21</th></tr>
<tr><td><b>Domain</b></td><td>Social network</td><td>Computer network traffic</td></tr>
<tr><td><b>Nodes</b></td><td>7,624 users</td><td>52 hosts</td></tr>
<tr><td><b>Edges</b></td><td>27,806 mutual friendships</td><td>1,275 directed communications</td></tr>
<tr><td><b>Directed</b></td><td>No</td><td>Yes</td></tr>
<tr><td><b>Density</b></td><td>0.096% (sparse)</td><td>96% raw / 51% filtered (dense)</td></tr>
<tr><td><b>Relation Types</b></td><td>1 (friendship)</td><td>2,157 (port/protocol)</td></tr>
<tr><td><b>Node Features</b></td><td>7,842 binary (artist preferences)</td><td>None (engineered from ports)</td></tr>
<tr><td><b>Ground Truth</b></td><td>18 country labels</td><td>23 functional groups</td></tr>
</table>

**Sources:**
- [LastFM Asia Social Network](https://snap.stanford.edu/data/feather-lastfm-social.html) — Rozemberczki & Sarkar, 2020
- [Cisco Networks](https://snap.stanford.edu/data/cisco-networks.html) — Cisco Secure Workload group

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`KamaliLassem_Mambelli.ipynb`](KamaliLassem_Mambelli.ipynb) | **Part 1** — Classical community detection (Louvain, Label Propagation, Greedy Modularity, Girvan-Newman, Infomap) on both datasets with intrinsic and external evaluation |
| [`KamaliLassem_Mambelli_Part2_progress.ipynb`](KamaliLassem_Mambelli_Part2_progress.ipynb) | **Part 2** — Spectral Clustering, Node2Vec, GCN (node classification & link prediction), TransE / RotatE / DistMult for link prediction, transfer learning (KGE→GCN), and comprehensive comparison |
| [`KamaliLassem_Mambelli_Presentation.ipynb`](KamaliLassem_Mambelli_Presentation.ipynb) | **Presentation** — Unified notebook: dataset introduction with visualizations, summary of Part 1 & early Part 2 methods with key plots, then detailed code for GCN, KGE models, transfer learning, and final comparison |

---

## Methods

### Part 1 — Classical Community Detection
- **Louvain** — Greedy modularity optimization
- **Label Propagation** — Iterative neighbor-majority voting
- **Greedy Modularity** — Agglomerative community merging
- **Girvan-Newman** — Divisive hierarchical (directed)
- **Infomap** — Information-theoretic random walks (directed)

### Part 2 — Embeddings & GNNs
- **Spectral Clustering** — Graph Laplacian eigenvectors + K-Means
- **Node2Vec** — Biased random walks + Word2Vec (BFS & DFS configurations)
- **GCN** — 2-layer Graph Convolutional Network for node classification and link prediction
- **TransE** — Translational knowledge graph embeddings
- **RotatE** — Rotational embeddings in complex space
- **DistMult** — Bilinear diagonal embeddings
- **Transfer Learning** — KGE entity embeddings as GCN input features

---

## Selected Results

### Community Detection on LastFM

| Method | Type | Modularity | NMI | ARI |
|--------|------|-----------|-----|-----|
| **Louvain** | Classical | **0.815** | 0.622 | 0.535 |
| Spectral (k=18) | Spectral | 0.799 | 0.659 | **0.671** |
| Node2Vec DFS | Shallow Emb. | 0.804 | 0.635 | 0.561 |
| **GCN (SVD+RotatE)** | GNN + KGE | 0.713 | **0.680** | 0.595 |

### Link Prediction — Cross-Dataset Comparison

| Model | LastFM MRR | LastFM Hits@10 | Cisco MRR | Cisco Hits@10 |
|-------|-----------|----------------|-----------|---------------|
| GCN (dot-product) | 0.058 | 12.82% | — | — |
| TransE | 0.073 | 21.8% | **0.828** | **97.6%** |
| **RotatE** | **0.121** | **31.8%** | 0.662 | 95.1% |
| DistMult | 0.095 | 21.7% | 0.452 | 89.2% |

### Transfer Learning: KGE → GCN Node Classification

| Variant | LastFM Accuracy | Cisco Accuracy |
|---------|----------------|----------------|
| GCN (original features) | 89.4% | 20.0% |
| **GCN (orig + DistMult)** | **90.0%** | 11.1% |
| GCN (orig + TransE) | 89.9% | **37.8%** |

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

**Python** ≥ 3.10 recommended. PyTorch should be installed with CUDA support if a GPU is available:

```bash
# Example for CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Data

- **LastFM:** Place the `lastfm_asia/` folder (containing `lastfm_asia_edges.csv`, `lastfm_asia_features.json`, `lastfm_asia_target.csv`) in the repo root
- **Cisco:** Place the `Cisco_22_networks/` folder in the repo root

Both datasets are available from the [SNAP](https://snap.stanford.edu/data/) repository.

---

## References

1. Von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing*, 17(4).
2. Grover, A. & Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." *KDD*.
3. Kipf, T. N. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR*.
4. Bordes, A. et al. (2013). "Translating Embeddings for Modeling Multi-relational Data." *NeurIPS*.
5. Sun, Z. et al. (2019). "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." *ICLR*.
6. Yang, B. et al. (2015). "Embedding Entities and Relations for Learning and Inference in Knowledge Bases." *ICLR*.
7. Rozemberczki, B. & Sarkar, R. (2020). "Characteristic Functions on Graphs: Birds of a Feather." *CIKM*.
8. Blondel, V. D. et al. (2008). "Fast unfolding of communities in large networks." *JSTAT*.
9. Ali, M. et al. (2021). "PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings." *JMLR*.

---

## License

This project is for educational purposes as part of the Massive Graph course at Polytechnique Montréal.
