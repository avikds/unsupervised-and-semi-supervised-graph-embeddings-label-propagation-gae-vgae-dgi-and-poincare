# Unsupervised and Semi-Supervised Graph Embeddings: Label Propagation, GAE/VGAE, DGI and Poincare

Murphy's Probabilistic Machine Learning claims one framework covers every graph embedding method: an encoder, a decoder, and a loss with three weights (GraphEDM). This project tests that, in tinygrad, on the methods the site's other GNN projects leave out, using graphs where the truth is known: a degree-corrected contextual stochastic block model that keeps its true edge probabilities, and a balanced tree with an exact metric. You write one trainer that is literally the GraphEDM loss, and every neural method is a call to it. Label propagation by iteration and in closed form, where one class swallows the graph at 1% labels until class mass normalisation repairs it, then label spreading. Laplacian eigenmaps, where one tiny component costs an embedding dimension. DeepWalk's similarity matrix, with its window weights derived and checked by simulation because the printed expression is not a distribution, then factorised as NetMF with no walk ever sampled. GAE and VGAE, including why an inner-product decoder cannot learn without reweighting positives, and what the KL weight really does. Deep Graph Infomax, judged by a linear probe against its own untrained encoder and against plain feature smoothing. A GCN classifier with an optional reconstruction term, and Planetoid. Poincare embeddings with Riemannian SGD, against Euclidean embeddings and classical MDS at equal dimension. Headline numbers, means over ten seeds on the prod CPU image with 1000 nodes and five classes. Accuracy at 2 and 10 labels per class: raw-feature probe 0.35 and 0.51, harmonic label propagation 0.28 and 0.69, with class mass normalisation 0.75 and 0.83, label spreading 0.61 and 0.79, Planetoid 0.59 and 0.68, GAE probe 0.69 and 0.84, untrained two-hop smoothing 0.82 and 0.92, DGI probe 0.85 and 0.92, GCN 0.83 and 0.93. Adding reconstruction to the GCN lowered accuracy on all ten seeds (0.83 to 0.66 at 2 labels, beta = 1). Joint Planetoid rarely beat its own two-stage pipeline (2 of 10 seeds at 2 labels, none at 10), so the book's motivation for it did not reproduce here. Held-out link AUC: the true probabilities 0.86, common neighbours 0.55, NetMF 0.69, GAE 0.72, VGAE 0.71 at the ELBO's own KL scale, 0.77 with the KL turned up 100 times, and 0.60 at 1000 times as the posterior nearly collapses. On a heterophilous graph label propagation falls below chance at 0.16, a GCN reaches 0.30, and a linear model on raw features 0.54. On a 364-node tree, mean average precision is 0.94 for 2-D Poincare, 0.68 for 2-D Euclidean, 0.79 for 10-D Euclidean and 0.04 for 2-D MDS, though Euclidean has the better mean rank from 5 dimensions up. All data is synthetic, with one Gaussian mean per class, so read these as results about this family of graphs. The assembled script runs the whole study for one seed in three to five minutes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** contextual_sbm
- [x] **2.** edge_homophily
- [x] **3.** balanced_tree
- [x] **4.** label_propagation
- [x] **5.** harmonic_labels
- [x] **6.** class_mass_normalize
- [x] **7.** label_spreading
- [x] **8.** kmeans
- [x] **9.** laplacian_eigenmaps
- [x] **10.** deepwalk_similarity
- [x] **11.** netmf_embedding
- [x] **12.** edm_fit
- [x] **13.** GCNEncoder
- [x] **14.** link_split
- [x] **15.** link_auc
- [x] **16.** gae_full_loss
- [x] **17.** train_gae
- [x] **18.** train_vgae
- [x] **19.** dgi_loss
- [x] **20.** train_dgi
- [x] **21.** linear_probe
- [x] **22.** train_gcn_classifier
- [x] **23.** sample_context_pairs
- [x] **24.** train_planetoid
- [x] **25.** poincare_distance
- [x] **26.** riemannian_update
- [x] **27.** train_tree_embedding
- [x] **28.** reconstruction_metrics
- [x] **29.** classical_mds
- [x] **30.** graph_embeddings_report

## Results

```
Main graph: 2469 edges, homophily 0.800, 26 isolated nodes, largest degree 46

1. Labels as embeddings: the flood at 1% labels (no degree correction, average degree 8, 2 labels per class)
   harmonic solution (eq 23.19)                       0.478   [0.32-0.59, mean 0.43]
   share of unlabeled nodes in its largest class      0.703   [0.45-0.87, mean 0.69]
   same F after class mass normalisation              0.918   [0.83-0.94, mean 0.90]
   label spreading (eq 23.20)                         0.873   [0.74-0.90, mean 0.84]

2. Community recovery with no labels: adjusted Rand index of K-means
   zero eigenvalues of the Laplacian on this seed: 1 (over ten seeds: 1 to 2)
   raw features                                       0.453   [0.43-0.54, mean 0.46]
   Laplacian eigenmaps (eq 23.8)                      0.596   [0.36-0.76, mean 0.60]
   eigenmaps, tau = 1                                 0.757   [0.72-0.76, mean 0.75]
   NetMF: DeepWalk factorised, no walks (eq 23.17-23.18) 0.739   [0.71-0.76, mean 0.74]
   two-hop smoothed features                          0.890   [0.66-0.89, mean 0.82]
   GAE embedding                                      0.787   [0.78-0.83, mean 0.79]
   untrained DGI encoder                              0.605   [0.45-0.74, mean 0.55]
   DGI embedding (eq 23.34)                           0.872   [0.60-0.88, mean 0.82]

3. Link prediction on held-out edges: AUC
   true edge probabilities (the ceiling)              0.865   [0.85-0.88, mean 0.86]
   common neighbours                                  0.539   [0.52-0.57, mean 0.55]
   eigenmaps                                          0.610   [0.57-0.62, mean 0.59]
   NetMF                                              0.699   [0.67-0.70, mean 0.69]
   GAE, negative sampling                             0.727   [0.70-0.75, mean 0.72]
   GAE, full loss with pos_weight (eq 23.31)          0.725   [0.69-0.75, mean 0.72]
   GAE, full loss with no weighting: stalls at log 2  0.573   [0.48-0.57, mean 0.54]
   GAE, identity instead of features                  0.686   [0.68-0.74, mean 0.71]
   GAE, no graph in the encoder                       0.675   [0.66-0.69, mean 0.67]
   VGAE (eq 23.32-23.33). kl_weight = 1 is about the ELBO's own scale; 1000 = N forgets the 1/N.
   kl_weight      1: AUC                              0.743   [0.68-0.76, mean 0.71]
   kl_weight      1: KL per node, nats (16 of 16 dims above 0.1) 35.854   [35.05-36.28, mean 35.68]
   kl_weight    100: AUC                              0.799   [0.75-0.80, mean 0.77]
   kl_weight    100: KL per node, nats (16 of 16 dims above 0.1) 5.010   [4.78-5.06, mean 4.93]
   kl_weight    300: AUC                              0.809   [0.75-0.81, mean 0.77]
   kl_weight    300: KL per node, nats (15 of 16 dims above 0.1) 1.999   [1.85-2.04, mean 1.98]
   kl_weight   1000: AUC                              0.598   [0.58-0.62, mean 0.60]
   kl_weight   1000: KL per node, nats (0 of 16 dims above 0.1) 0.336   [0.30-0.35, mean 0.33]

4. Node classification: accuracy on unlabeled nodes
   2 labels per class
   probe on raw features                              0.322   [0.28-0.42, mean 0.35]
   label propagation                                  0.286   [0.21-0.40, mean 0.28]
   label propagation + class mass normalisation       0.824   [0.68-0.82, mean 0.75]
   label spreading                                    0.673   [0.51-0.73, mean 0.61]
   probe on context-only Zc (two-stage)               0.715   [0.43-0.71, mean 0.62]
   Planetoid, joint (eq 23.35-23.36)                  0.605   [0.51-0.67, mean 0.59]
   probe on GAE                                       0.779   [0.54-0.78, mean 0.69]
   probe on untrained DGI encoder                     0.686   [0.58-0.74, mean 0.66]
   probe on DGI                                       0.906   [0.74-0.91, mean 0.85]
   probe on two-hop smoothing, nothing trained        0.879   [0.72-0.88, mean 0.82]
   GCN classifier                                     0.892   [0.71-0.89, mean 0.83]
   GCN + reconstruction, beta = 0.1                   0.769   [0.61-0.77, mean 0.72]
   GCN + reconstruction, beta = 1.0                   0.726   [0.53-0.73, mean 0.66]
   10 labels per class
   probe on raw features                              0.436   [0.44-0.55, mean 0.51]
   label propagation                                  0.724   [0.51-0.81, mean 0.69]
   label propagation + class mass normalisation       0.821   [0.80-0.85, mean 0.83]
   label spreading                                    0.776   [0.74-0.81, mean 0.79]
   probe on context-only Zc (two-stage)               0.778   [0.68-0.79, mean 0.75]
   Planetoid, joint (eq 23.35-23.36)                  0.617   [0.58-0.73, mean 0.68]
   probe on GAE                                       0.854   [0.79-0.89, mean 0.84]
   probe on untrained DGI encoder                     0.861   [0.82-0.90, mean 0.87]
   probe on DGI                                       0.916   [0.91-0.94, mean 0.92]
   probe on two-hop smoothing, nothing trained        0.915   [0.90-0.93, mean 0.92]
   GCN classifier                                     0.922   [0.91-0.94, mean 0.93]
   GCN + reconstruction, beta = 0.1                   0.875   [0.87-0.90, mean 0.89]
   GCN + reconstruction, beta = 1.0                   0.842   [0.83-0.89, mean 0.86]
   Reconstruction lowered GCN accuracy on all ten seeds at both budgets.
   Joint Planetoid beat its own two-stage probe on 2 of 10 seeds at 2 labels and 0 of 10 at 10: the book's motivation did not reproduce here.
   DGI, GCN and plain smoothing are within the spread of each other at both budgets.

5. A graph that breaks the assumption: homophily 0.058, 10 labels per class, chance 0.2
   label propagation                                  0.160   [0.15-0.17, mean 0.16]
   label spreading                                    0.166   [0.15-0.18, mean 0.17]
   probe on two-hop smoothing                         0.266   [0.27-0.34, mean 0.31]
   GCN classifier                                     0.271   [0.27-0.33, mean 0.30]
   probe on raw features                              0.484   [0.48-0.58, mean 0.54]
   ARI, eigenmaps                                     -0.001   [about 0 on all ten seeds]
   ARI, raw features                                  0.440   [0.27-0.56, mean 0.46]

6. Reconstructing a 364-node tree (3-ary, depth 5)
                               MAP  mean rank  distortion
   poincare d=2              0.937       3.52       0.195   [MAP 0.93-0.95, mean 0.94]
   poincare d=5              0.972       1.92       0.195   [MAP 0.97-0.99, mean 0.98]
   euclidean d=2             0.685       5.97       0.400   [MAP 0.66-0.71, mean 0.68]
   euclidean d=5             0.762       1.33       0.206   [MAP 0.75-0.77, mean 0.76]
   euclidean d=10            0.791       1.21       0.180   [MAP 0.79-0.80, mean 0.79]
   mds d=2                   0.040      69.04       0.548   [MAP 0.04-0.04, mean 0.04]
   mds d=5                   0.102      22.10       0.276   [MAP 0.10-0.10, mean 0.10]
   mds d=10                  0.171      11.15       0.147   [MAP 0.16-0.16, mean 0.16]
   Poincare d=2: Euclidean norm of the root           0.916   [0.55-0.96, mean 0.82]
   Poincare d=2: corr(distance to root, tree level)   0.756   [0.75-0.85, mean 0.79]
   Poincare wins MAP at every dimension tried and mean rank at d=2. At d=5 Euclidean has the better mean rank.
   MDS at d=10 has the lowest distortion of all, yet poor local ranking. The loss does not put the root at the origin.
   MDS has no seed, but repeated eigenvalues can move its scores slightly from one machine to another.

7. The GraphEDM table (eq 23.4): L = alpha SUP + beta RECON + gamma REG
   method               encoder          target s(W)              decoder               loss                     alpha  beta  gamma         result
   MDS                  Spectral decomposition Pairwise distances       Euclidean distance    Classical distance reconstruction 0      0     0             tree MAP 0.04 at d=2
   Laplacian eigenmaps  Graph Laplacian eigenvectors Graph structure          Euclidean distance    Laplacian energy         0      0     0             ARI 0.76 with tau
   Poincare embedding   Learned node embeddings Tree structure           Poincare distance     Tree softmax             0      1     0             tree MAP 0.94 at d=2
   Label propagation    Identity         Node labels              identity              Harmonic energy          1      0     0             acc 0.72 at 10/class
   DeepWalk             Node embeddings  Random walk contexts     Dot product           Context reconstruction   0      1     0             analytical context similarity; no separate score reported
   NetMF                Matrix factorisation DeepWalk matrix          Dot product           Matrix factorisation     0      1     0             ARI 0.74, AUC 0.70
   GAE                  GCN              Adjacency matrix         Inner product         Reconstruction BCE       0      1     0             AUC 0.73, ARI 0.79
   VGAE                 Variational GCN  Adjacency matrix         Inner product         Reconstruction BCE + KL  0      1     KL            AUC 0.80 at kl_weight 100
   DGI                  GCN              Graph summary            Bilinear discriminator Contrastive information loss 0      1     0             probe 0.92 at 10/class
   GCN classifier       GCN              Node labels              Softmax               Cross-entropy            1      0     1             acc 0.92 at 10/class
   Planetoid            Feature and context lookup Labels and context pairs Softmax + context dot product Classification + context reconstruction 1      1     1             acc 0.62 at 10/class

study finished in 555 s
```
