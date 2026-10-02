#!/usr/bin/env python3
"""
scanpy_pipeline.py
===================
Complete single-cell RNA-seq analysis pipeline built on scanpy, for data
supplied as a CSV/TSV count matrix (genes x cells OR cells x genes).

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

Usage
-----
    # Run on your own data
    python scanpy_pipeline.py --input counts.csv --transpose \
        --batch-key batch --species human --resolution 1.0 \
        --output-dir results/

    # Try it out with an auto-generated synthetic demo dataset
    python scanpy_pipeline.py --demo --output-dir results_demo/

Input format
------------
--input should be a CSV/TSV file. By default rows = cells, columns = genes
(the standard scanpy/AnnData orientation). If your file is genes x cells
(common for bulk-style exports, e.g. 10x CSV exports), pass --transpose.
The first column/row is used as the index (barcodes/gene names).

If you have a batch/sample/donor column you'd like to integrate across,
either:
  (a) supply --batch-key referencing a column already in your matrix, or
  (b) supply a separate --metadata CSV (cell_id + batch + any other obs
      columns) via --metadata / --metadata-batch-col.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, facecolor="white")


# ---------------------------------------------------------------------------
# 0. Demo data generator (so the pipeline is runnable end-to-end with no
#    external downloads / internet access)
# ---------------------------------------------------------------------------
def generate_demo_csv(path, n_cells=1200, n_genes=800, n_batches=3, seed=0):
    """Create a synthetic cells x genes CSV with 4 cell-type-like blobs
    spread across multiple batches, plus mitochondrial-like genes, so the
    full pipeline (QC -> integration -> clustering -> annotation) has
    something realistic to chew on."""
    rng = np.random.default_rng(seed)
    n_types = 4
    cells_per_type = n_cells // n_types

    gene_names = [f"GENE{i}" for i in range(n_genes - 13)]
    mt_genes = [f"MT-{g}" for g in ["ND1", "ND2", "CO1", "CO2", "ATP8",
                                     "ATP6", "CO3", "ND3", "ND4L", "ND4",
                                     "ND5", "ND6", "CYB"]]
    genes = gene_names + mt_genes

    # give each cell type a distinct block of "marker" genes with high mean
    marker_block = n_genes // n_types
    counts = np.zeros((n_cells, n_genes), dtype=int)
    cell_types = []
    batches = []

    for t in range(n_types):
        start = t * cells_per_type
        end = n_cells if t == n_types - 1 else start + cells_per_type
        base = rng.negative_binomial(3, 0.5, size=(end - start, n_genes))
        marker_start, marker_end = t * marker_block, (t + 1) * marker_block
        base[:, marker_start:marker_end] += rng.negative_binomial(
            15, 0.3, size=(end - start, marker_end - marker_start)
        )
        counts[start:end] = base
        cell_types += [f"CellType{t}"] * (end - start)
        batches += list(rng.integers(0, n_batches, size=(end - start,)))

    # mild batch effect: scale a random subset of genes per batch
    for b in range(n_batches):
        mask = np.array(batches) == b
        shift = rng.normal(1.0, 0.15, size=n_genes)
        counts[mask] = (counts[mask] * shift).astype(int)
    counts = np.clip(counts, 0, None)

    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    df = pd.DataFrame(counts, index=cell_ids, columns=genes)
    df.insert(0, "batch", [f"batch{b}" for b in batches])
    df.to_csv(path)
    print(f"[demo] wrote synthetic dataset with true labels embedded "
          f"(cell types hidden) -> {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path, transpose=False, batch_col_in_matrix=None):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, index_col=0)

    obs_extra = None
    if batch_col_in_matrix and batch_col_in_matrix in df.columns:
        obs_extra = df[[batch_col_in_matrix]].copy()
        df = df.drop(columns=[batch_col_in_matrix])

    # drop any remaining non-numeric columns defensively
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[load] dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    if transpose:
        df = df.T  # now cells (rows) x genes (cols)

    adata = ad.AnnData(X=df.values.astype(np.float32))
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns.astype(str)
    adata.var_names_make_unique()

    if obs_extra is not None:
        obs_extra = obs_extra.loc[df.index] if not transpose else obs_extra
        for col in obs_extra.columns:
            adata.obs[col] = obs_extra[col].values

    print(f"[load] AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def attach_metadata(adata, metadata_path, batch_col):
    meta = pd.read_csv(metadata_path, index_col=0)
    meta.index = meta.index.astype(str)
    common = adata.obs_names.intersection(meta.index)
    if len(common) == 0:
        print("[metadata] WARNING: no overlapping cell IDs found; skipping")
        return adata
    meta = meta.loc[adata.obs_names.intersection(meta.index)]
    for col in meta.columns:
        adata.obs[col] = meta[col].reindex(adata.obs_names)
    if batch_col not in adata.obs.columns:
        print(f"[metadata] WARNING: batch column '{batch_col}' not found "
              "after merge")
    return adata


# ---------------------------------------------------------------------------
# 2. QC + filtering
# ---------------------------------------------------------------------------
def qc_filter(adata, species="human", min_genes=200, min_cells=3,
              max_pct_mt=20.0, max_genes=None, doublet_detection=True):
    mt_prefix = "MT-" if species == "human" else "mt-"
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-") \
        if species == "human" else adata.var_names.str.startswith(mt_prefix)

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)

    print("[qc] pre-filter:", adata.shape)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()
    if max_genes:
        adata = adata[adata.obs["n_genes_by_counts"] < max_genes].copy()
    print("[qc] post-filter:", adata.shape)

    if doublet_detection and adata.n_obs > 50:
        try:
            sc.pp.scrublet(adata, verbose=False)
            n_doublets = int(adata.obs["predicted_doublet"].sum())
            adata = adata[~adata.obs["predicted_doublet"]].copy()
            print(f"[qc] removed {n_doublets} predicted doublets -> "
                  f"{adata.shape}")
        except Exception as e:
            print(f"[qc] doublet detection skipped ({e})")

    return adata


# ---------------------------------------------------------------------------
# 3. Normalize + HVG
# ---------------------------------------------------------------------------
def normalize(adata, n_top_genes=2000, batch_key=None):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full log-normalized data for marker gene tests

    hvg_kwargs = dict(n_top_genes=n_top_genes, flavor="seurat")
    if batch_key and batch_key in adata.obs.columns:
        hvg_kwargs["batch_key"] = batch_key
    sc.pp.highly_variable_genes(adata, **hvg_kwargs)
    print(f"[normalize] selected {adata.var['highly_variable'].sum()} HVGs")
    return adata


# ---------------------------------------------------------------------------
# 4. Scale + PCA
# ---------------------------------------------------------------------------
def run_pca(adata, n_comps=50, max_value=10):
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=max_value)
    sc.tl.pca(adata_hvg, n_comps=n_comps, svd_solver="arpack")
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns["pca"]
    adata.varm = adata_hvg.varm if hasattr(adata_hvg, "varm") else adata.varm
    print(f"[pca] computed {n_comps} PCs on HVGs")
    return adata


# ---------------------------------------------------------------------------
# 5. Batch integration (Harmony)
# ---------------------------------------------------------------------------
def batch_integrate(adata, batch_key):
    if not batch_key or batch_key not in adata.obs.columns:
        print("[integration] no valid batch key provided; skipping Harmony")
        return adata, "X_pca"
    if adata.obs[batch_key].nunique() < 2:
        print("[integration] only one batch present; skipping Harmony")
        return adata, "X_pca"

    # Called directly against harmonypy (rather than via
    # sc.external.pp.harmony_integrate) because harmonypy >=2.0 changed the
    # orientation of Z_corr, which breaks scanpy's wrapper. Handling both
    # possible orientations here keeps this robust across harmonypy versions.
    import harmonypy
    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, [batch_key])
    z = np.asarray(ho.Z_corr)
    n_obs = adata.n_obs
    if z.shape[0] == n_obs:
        adjusted = z
    elif z.shape[1] == n_obs:
        adjusted = z.T
    else:
        raise ValueError(f"Unexpected Harmony output shape {z.shape} for "
                          f"{n_obs} cells")
    adata.obsm["X_pca_harmony"] = adjusted
    print(f"[integration] ran Harmony on batch key '{batch_key}'")
    return adata, "X_pca_harmony"


# ---------------------------------------------------------------------------
# 6. Neighbors, UMAP, Leiden
# ---------------------------------------------------------------------------
def cluster(adata, use_rep="X_pca", resolution=1.0, n_neighbors=15):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden",
                 flavor="igraph", n_iterations=2)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"[cluster] Leiden found {n_clusters} clusters at "
          f"resolution={resolution}")
    return adata


# ---------------------------------------------------------------------------
# 7. Marker genes
# ---------------------------------------------------------------------------
def find_markers(adata, groupby="leiden", n_genes=25, method="wilcoxon"):
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method,
                             n_genes=n_genes)
    groups = adata.obs[groupby].cat.categories
    rows = []
    for grp in groups:
        names = adata.uns["rank_genes_groups"]["names"][grp]
        lfc = adata.uns["rank_genes_groups"]["logfoldchanges"][grp]
        pvals = adata.uns["rank_genes_groups"]["pvals_adj"][grp]
        for gene, l, p in zip(names, lfc, pvals):
            rows.append({"cluster": grp, "gene": gene,
                          "log2fc": l, "pval_adj": p})
    marker_df = pd.DataFrame(rows)
    print(f"[markers] top marker table has {len(marker_df)} rows across "
          f"{len(groups)} clusters")
    return adata, marker_df


# ---------------------------------------------------------------------------
# 8. Simple marker-score-based annotation
# ---------------------------------------------------------------------------
DEFAULT_MARKERS_HUMAN = {
    "T cell": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
    "B cell": ["CD79A", "CD79B", "MS4A1", "CD19"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "Monocyte/Myeloid": ["LYZ", "CD14", "FCGR3A", "CST3", "ITGAM"],
    "Dendritic cell": ["FCER1A", "CLEC9A", "ITGAX"],
    "Platelet": ["PPBP", "PF4"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
}


def annotate_clusters(adata, marker_dict=None, groupby="leiden"):
    """Score each cell against canonical marker sets (sc.tl.score_genes),
    then assign each cluster the cell type with the highest mean score.
    This is a fast, transparent default -- for production work, replace
    marker_dict with curated markers for your tissue, or swap in a
    reference-based tool (e.g. CellTypist, scANVI, SingleR-style mapping).
    """
    marker_dict = marker_dict or DEFAULT_MARKERS_HUMAN
    score_cols = []
    for cell_type, genes in marker_dict.items():
        genes_present = [g for g in genes if g in adata.raw.var_names] \
            if adata.raw is not None else [g for g in genes if g in adata.var_names]
        if len(genes_present) == 0:
            continue
        col = f"score_{cell_type}"
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=col,
                           use_raw=adata.raw is not None)
        score_cols.append((col, cell_type))

    if not score_cols:
        print("[annotate] none of the marker genes were found in this "
              "dataset -- skipping annotation. Supply a custom marker_dict "
              "matching your gene naming (e.g. mouse gene symbols).")
        adata.obs["cell_type"] = "unknown"
        return adata, pd.DataFrame()

    score_df = adata.obs[[c for c, _ in score_cols]].copy()
    cluster_means = score_df.groupby(adata.obs[groupby], observed=True).mean()
    cluster_to_type = cluster_means.idxmax(axis=1).map(
        {c: t for c, t in score_cols}
    )
    adata.obs["cell_type"] = adata.obs[groupby].map(cluster_to_type).astype(str)
    print("[annotate] cluster -> cell type assignment:")
    for cl, ct in cluster_to_type.items():
        print(f"    cluster {cl}: {ct}")
    return adata, cluster_means


# ---------------------------------------------------------------------------
# 9. Plots + saving
# ---------------------------------------------------------------------------
def make_plots(adata, outdir, groupby="leiden"):
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    sc.settings.figdir = figdir

    sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                 jitter=0.4, multi_panel=True, save="_qc.png", show=False)
    sc.pl.umap(adata, color=[groupby], save="_leiden.png", show=False)
    if "cell_type" in adata.obs.columns:
        sc.pl.umap(adata, color=["cell_type"], save="_celltype.png", show=False)
    if "batch" in adata.obs.columns or any(
        c for c in adata.obs.columns if "batch" in c.lower()
    ):
        batch_col = next(c for c in adata.obs.columns if "batch" in c.lower())
        sc.pl.umap(adata, color=[batch_col], save="_batch.png", show=False)
    print(f"[plots] saved figures to {figdir}")


def save_outputs(adata, marker_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    h5ad_path = os.path.join(outdir, "adata_processed.h5ad")
    adata.write_h5ad(h5ad_path)

    marker_path = os.path.join(outdir, "marker_genes_per_cluster.csv")
    marker_df.to_csv(marker_path, index=False)

    summary_cols = [c for c in ["leiden", "cell_type"] if c in adata.obs.columns]
    summary = adata.obs[summary_cols].value_counts().reset_index(name="n_cells") \
        if summary_cols else pd.DataFrame()
    summary_path = os.path.join(outdir, "cluster_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[save] wrote: {h5ad_path}, {marker_path}, {summary_path}")
    return h5ad_path, marker_path, summary_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="Path to CSV/TSV count matrix")
    p.add_argument("--transpose", action="store_true",
                    help="Pass if input is genes (rows) x cells (cols)")
    p.add_argument("--batch-key", default=None,
                    help="obs column name to integrate across with Harmony "
                         "(either a column already in the matrix, or a "
                         "column provided via --metadata)")
    p.add_argument("--metadata", default=None,
                    help="Optional separate CSV of cell_id + batch/other obs columns")
    p.add_argument("--species", default="human", choices=["human", "mouse"])
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--max-pct-mt", type=float, default=20.0)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--no-doublet-detection", action="store_true")
    p.add_argument("--output-dir", default="scanpy_results")
    p.add_argument("--demo", action="store_true",
                    help="Ignore --input and run on an auto-generated "
                         "synthetic demo dataset (no internet required)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo:
        demo_path = os.path.join(args.output_dir, "demo_counts.csv")
        generate_demo_csv(demo_path)
        args.input = demo_path
        args.batch_key = args.batch_key or "batch"
        args.transpose = False
    elif not args.input:
        p.error("--input is required unless --demo is set")

    # 1. load
    adata = load_data(args.input, transpose=args.transpose,
                       batch_col_in_matrix=args.batch_key)
    if args.metadata:
        adata = attach_metadata(adata, args.metadata, args.batch_key)

    # 2. QC
    adata = qc_filter(adata, species=args.species, min_genes=args.min_genes,
                       min_cells=args.min_cells, max_pct_mt=args.max_pct_mt,
                       doublet_detection=not args.no_doublet_detection)

    # 3. normalize + HVG
    adata = normalize(adata, n_top_genes=args.n_top_genes,
                       batch_key=args.batch_key)

    # 4. PCA
    adata = run_pca(adata, n_comps=args.n_pcs)

    # 5. batch integration
    adata, rep = batch_integrate(adata, args.batch_key)

    # 6. cluster
    adata = cluster(adata, use_rep=rep, resolution=args.resolution)

    # 7. markers
    adata, marker_df = find_markers(adata, groupby="leiden")

    # 8. annotate
    marker_dict = DEFAULT_MARKERS_HUMAN if args.species == "human" else None
    adata, _ = annotate_clusters(adata, marker_dict=marker_dict, groupby="leiden")

    # 9. plots + save
    make_plots(adata, args.output_dir, groupby="leiden")
    save_outputs(adata, marker_df, args.output_dir)

    print("\n[done] pipeline complete. Outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
