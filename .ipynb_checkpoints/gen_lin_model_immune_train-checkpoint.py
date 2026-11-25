import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TheilSenRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import warnings
from multiprocessing import Pool
from src.simulate_func import lin_model_sample,parallel_sim,gen_lin_model,parallel_gen_lin_model

num_samples = 2 # 100
num_cores = 24

# Get transcriptomics and proteomics data from dataset USED FOR TRAINING
df = pd.read_csv("data/PBMCs_dataset.csv").dropna()

# Only get genes with enough data and that are differentially expressed 
var_TPM = df[["Gene","Log2 TPM"]].groupby("Gene").std()
var_TPM = var_TPM.loc[var_TPM["Log2 TPM"]>=var_TPM["Log2 TPM"].median()].index
Num_data = df[["Gene","Log2 TPM"]].groupby("Gene").count()
Num_data = Num_data.loc[Num_data["Log2 TPM"]>=int(Num_data["Log2 TPM"].max()/2)].index
gene_list = list(set(var_TPM) & set(Num_data))
print("Number of genes studied: "+str(len(gene_list)))
df = df.loc[df["Gene"].isin(gene_list)]

# Separate data into training and test X times by removing 2 random cell types (around 80/20 split)
df_train = pd.DataFrame(columns=list(df.columns)+["Sample"])
df_test = pd.DataFrame(columns=list(df.columns)+["Sample"])
warnings.filterwarnings('ignore')
for i in range(0,num_samples):
    test_cells = df["Cell"].sample(n=2).values
    df_train_slice = df.loc[~df["Cell"].isin(test_cells)]
    df_test_slice = df.loc[df["Cell"].isin(test_cells)]
    df_train_slice["Sample"] = i
    df_test_slice["Sample"] = i
    df_train = pd.concat([df_train, df_train_slice], ignore_index=True)
    df_test = pd.concat([df_test, df_test_slice], ignore_index=True)
warnings.filterwarnings('default')
print("Data sampling done!")

# Generate linear model for all samples
warnings.filterwarnings('ignore')
df_lin = parallel_gen_lin_model(df_train, gene_list, num_samples, num_cores)
warnings.filterwarnings('default')
df_lin.to_csv('results/lin_model/linear_immune.tsv.gz', sep='\t', index=False, compression='gzip')
print("Linear models done!")

# Simulate for the training set of each sample
df_train["Prot. count predict"] = np.nan
df_train = parallel_sim(df_train, df_lin, gene_list, num_samples, num_cores)
df_train.to_csv('results/lin_model/linear_immune_train.tsv.gz', sep='\t', index=False, compression='gzip')
print("Training set simulation done! (PBMCs)")

# Simulate for the validation set of each sample
df_test["Prot. count predict"] = np.nan
df_test = parallel_sim(df_test, df_lin, gene_list, num_samples, num_cores)
df_test.to_csv('results/lin_model/linear_immune_validation.tsv.gz', sep='\t', index=False, compression='gzip')
print("Validation set simulation done! (PBMCs)")

# Test models with data from different source (tissue data)
df_test_tissues = pd.read_csv("data/tissues_dataset.csv")
df_test_tissues_sample = df_test_tissues.loc[df_test_tissues["Gene"].isin(gene_list)]
warnings.filterwarnings('ignore')
df_test_tissues = pd.DataFrame(columns= df_test_tissues_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_tissues_sample.copy()
    df_sample["Sample"] = i
    df_test_tissues = pd.concat([df_test_tissues, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set 1 for each sample
df_test_tissues["Prot. count predict"] = np.nan
df_test_tissues = parallel_sim(df_test_tissues, df_lin, gene_list, num_samples, num_cores)
df_test_tissues.to_csv('results/lin_model/linear_immune_test_tissues.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (tissues)")

# Test models with data from different source (cell lines data)
df_test_cell_lines = pd.read_csv("data/cell_lines_dataset.csv")
df_test_cell_lines_sample = df_test_cell_lines.loc[df_test_cell_lines["Gene"].isin(gene_list)]
warnings.filterwarnings('ignore')
df_test_cell_lines = pd.DataFrame(columns= df_test_cell_lines_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_cell_lines_sample.copy()
    df_sample["Sample"] = i
    df_test_cell_lines = pd.concat([df_test_cell_lines, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set 2 for each sample
df_test_cell_lines["Prot. count predict"] = np.nan
df_test_cell_lines = parallel_sim(df_test_cell_lines, df_lin, gene_list, num_samples, num_cores)
df_test_cell_lines.to_csv('results/lin_model/linear_immune_test_cell_lines.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (cell lines)")

# Test models with data from different source (erythropoiesis data)
df_test_erythr = pd.read_csv("data/erythropoiesis_dataset.csv")
df_test_erythr_sample = df_test_erythr.loc[df_test_erythr["Gene"].isin(gene_list)]
warnings.filterwarnings('ignore')
df_test_erythr = pd.DataFrame(columns= df_test_erythr_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_erythr_sample.copy()
    df_sample["Sample"] = i
    df_test_erythr = pd.concat([df_test_erythr, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set 3 for each sample
df_test_erythr["Prot. count predict"] = np.nan
df_test_erythr = parallel_sim(df_test_erythr, df_lin, gene_list, num_samples, num_cores)
df_test_erythr.to_csv('results/lin_model/linear_immune_test_erythropoiesis.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (erythropoiesis)")