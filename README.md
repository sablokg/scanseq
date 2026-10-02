# scanseq

- my scanpy based complete analysis build over reading.
- Complete single-cell RNA-seq analysis pipeline built on scanpy, for data supplied as a CSV/TSV count matrix (genes x cells OR cells x genes).

```
Pipeline stages
----------------
1. Load raw counts from CSV/TSV -> AnnData
2. QC metrics + filtering (min genes/cell, min cells/gene, %mito, doublets)
3. Normalization (total-count + log1p) and highly variable gene selection
4. Scaling + PCA
5. Batch integration with Harmony (if a batch column is provided)
6. Neighbors graph, UMAP embedding, Leiden clustering
7. Marker gene detection per cluster (rank_genes_groups)
8. Simple marker-score-based cell type annotation (editable marker dictionary)
9. Save results: annotated .h5ad, marker gene table, cluster summary, plots
```

Gaurav Sablok \
gsablok@proton.me
