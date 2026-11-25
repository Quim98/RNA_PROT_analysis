import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_absolute_error

# Metrics averaged per sample. Columns that are not produced by
# calculate_gene_metrics (e.g. "ME", "Spearman RNA") are skipped automatically,
# so adding a metric there is enough for it to show up in the per-sample means.
METRIC_COLS = ["R2", "MAE", "ME", "Spearman prediction", "Spearman RNA"]
VALUE_COLS = ["Log10 Prot. count", "Prot. count predict", "Log2 TPM"]


def calculate_gene_metrics(group):
    pred = group[["Log10 Prot. count", "Prot. count predict"]].dropna()

    return pd.Series({
        "R2": (
            r2_score(pred["Log10 Prot. count"], pred["Prot. count predict"])
            if len(pred) >= 2 else np.nan
        ),
        "MAE": (
            mean_absolute_error(pred["Log10 Prot. count"], pred["Prot. count predict"])
            if len(pred) >= 1 else np.nan
        ),
        "Spearman prediction": (
            pred["Log10 Prot. count"].corr(pred["Prot. count predict"], method="spearman")
            if len(pred) >= 2 else np.nan
        ),
    })


def _metrics_one_dataset(directory_data, dataset_name):
    """Worker: read one dataset and compute metrics per (Sample, Gene)."""
    df = pd.read_csv(directory_data + dataset_name + ".tsv.gz", sep="\t")
    cell_metrics = (
        df.groupby(["Sample", "Gene"])[VALUE_COLS]
        .apply(calculate_gene_metrics)
        .reset_index()
    )
    cell_metrics.insert(0, "Dataset", dataset_name)
    return cell_metrics


def _sample_means(cell_metrics):
    """Mean of each metric across genes, for each sample."""
    cols = [c for c in METRIC_COLS if c in cell_metrics.columns]
    sample_metrics = cell_metrics.groupby("Sample", as_index=False)[cols].mean()
    sample_metrics.insert(0, "Dataset", cell_metrics["Dataset"].iloc[0])
    return sample_metrics


def _compute_jobs(jobs, n_jobs=-1, verbose=5):
    """
    Run _metrics_one_dataset for every (directory_data, dataset_name) in `jobs`
    in parallel. Largest files are dispatched first so the biggest datasets don't
    end up as stragglers at the end. Results come back in the original order.
    """
    sizes = [os.path.getsize(d + name + ".tsv.gz") for d, name in jobs]
    order = sorted(range(len(jobs)), key=lambda i: -sizes[i])

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_metrics_one_dataset)(*jobs[i]) for i in order
    )

    out = [None] * len(jobs)
    for i, res in zip(order, results):
        out[i] = res
    return out


def metrics_per_gene(directory_data, data_list, n_jobs=-1, verbose=5):
    """Drop-in replacement for the original: parallel across the datasets of ONE model."""
    cell = _compute_jobs([(directory_data, d) for d in data_list], n_jobs, verbose)
    sample = [_sample_means(c) for c in cell]
    return pd.concat(cell, ignore_index=True), pd.concat(sample, ignore_index=True)


def metrics_all_models(models, out_dir="results/model_perf/per_cell/", n_jobs=-1, verbose=5):
    """
    Fastest option: runs ALL datasets of ALL models in one parallel pool, then writes
    f"{out_dir}{name}_gene.tsv.gz" and f"{out_dir}{name}_gene_mean.tsv.gz" for each model.

    models : {output name: (directory_data, data_list)}
    Returns {output name: (df_gene, df_mean)}.
    """
    jobs = [(directory, d) for directory, data_list in models.values() for d in data_list]
    cell = _compute_jobs(jobs, n_jobs, verbose)

    results, k = {}, 0
    for name, (_, data_list) in models.items():
        cell_model = cell[k:k + len(data_list)]
        k += len(data_list)
        df_gene = pd.concat(cell_model, ignore_index=True)
        df_mean = pd.concat([_sample_means(c) for c in cell_model], ignore_index=True)
        df_gene.to_csv(f"{out_dir}{name}_gene.tsv.gz", sep="\t", index=False, compression="gzip")
        df_mean.to_csv(f"{out_dir}{name}_gene_mean.tsv.gz", sep="\t", index=False, compression="gzip")
        results[name] = (df_gene, df_mean)
    return results


# ----------------------------------------------------------------------------
# Your four runs, now in a single parallel pool
# ----------------------------------------------------------------------------
cte_list = ["cte_immune_train", "cte_immune_validation", "cte_immune_test_tissues",
            "cte_immune_test_cell_lines", "cte_immune_test_erythropoiesis",
            "cte_tissues_train", "cte_tissues_validation", "cte_tissues_test_immune",
            "cte_tissues_test_cell_lines", "cte_tissues_test_erythropoiesis"]
lin_list = ["linear_immune_train", "linear_immune_validation", "linear_immune_test_tissues",
            "linear_immune_test_cell_lines", "linear_immune_test_erythropoiesis",
            "linear_tissues_train", "linear_tissues_validation", "linear_tissues_test_immune",
            "linear_tissues_test_cell_lines", "linear_tissues_test_erythropoiesis"]

MODELS = {
    "cte_model":                  ("results/cte_model/",                  cte_list),
    "lin_model":                  ("results/lin_model/",                  lin_list),
    "lin_model_no_gene_specific": ("results/lin_model/no_gene_specific/", lin_list),
    "lin_model_rand":             ("results/rand_lin_model/",             lin_list),
}

MODELS = {
    "lin_model_no_gene_specific": ("results/lin_model/no_gene_specific/", lin_list),
}

if __name__ == "__main__":
    metrics_all_models(MODELS, out_dir="results/model_perf/per_cell/")