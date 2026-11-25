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


num_samples = 100
num_cores = 24

# Load training and test data from linear models + slope/intersect dataframe
df_train = pd.read_csv('results/lin_model/linear_immune_train.tsv.gz', sep='\t')
df_train = df_train[['Dataset type','RNA dataset num.','Prot. count dataset num.','Cell','Gene','Log2 TPM','Log10 Prot. count','Sample']]
df_test = pd.read_csv('results/lin_model/linear_immune_validation.tsv.gz', sep='\t')
df_test = df_test[['Dataset type','RNA dataset num.','Prot. count dataset num.','Cell','Gene','Log2 TPM','Log10 Prot. count','Sample']]
df_lin = pd.read_csv('results/lin_model/linear_immune.tsv.gz', sep='\t')
gene_list = df_train["Gene"].drop_duplicates().to_list()
print("Linear models loaded!")

# Permutate the gene names
for i in df_lin["Sample"].drop_duplicates():
    df_lin.loc[df_lin["Sample"]==i,"Gene"] = np.random.permutation(df_lin.loc[df_lin["Sample"]==i,"Gene"].to_list())

# Simulate for the training set of each sample
df_train["Prot. count predict"] = np.nan
df_train["Log10 Prot. count"] = df_train["Log10 Prot. count"].replace(0,np.nan)
df_train = parallel_sim(df_train, df_lin, gene_list, num_samples, num_cores)
df_train.to_csv('results/rand_lin_model/linear_immune_train.tsv.gz', sep='\t', index=False, compression='gzip')
print("Training set simulation done!")

# Simulate for the validation set of each sample
df_test["Prot. count predict"] = np.nan
df_test["Log10 Prot. count"] = df_test["Log10 Prot. count"].replace(0,np.nan)
df_test = parallel_sim(df_test, df_lin, gene_list, num_samples, num_cores)
df_test.to_csv('results/rand_lin_model/linear_immune_validation.tsv.gz', sep='\t', index=False, compression='gzip')
print("Validation set simulation done!")

# Test models with data from different source (tissue data)
df_test_tissues = pd.read_csv("data/tissues_dataset.csv").dropna()
df_test_tissues_sample = df_test_tissues.loc[df_test_tissues["Gene"].isin(gene_list)]
df_test_tissues_sample["Log10 Prot. count"] = df_test_tissues_sample["Log10 Prot. count"].replace(0,np.nan)
warnings.filterwarnings('ignore')
df_test_tissues = pd.DataFrame(columns= df_test_tissues_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_tissues_sample.copy()
    df_sample["Sample"] = i
    df_test_tissues = pd.concat([df_test_tissues, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set for each sample
df_test_tissues["Prot. count predict"] = np.nan
df_test_tissues = parallel_sim(df_test_tissues, df_lin, gene_list, num_samples, num_cores)
df_test_tissues.to_csv('results/rand_lin_model/linear_immune_test_tissues.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (tissues)")

# Test models with data from different source (cell lines data)
df_test_cell_lines = pd.read_csv("data/cell_lines_dataset.csv")
df_test_cell_lines_sample = df_test_cell_lines.loc[df_test_cell_lines["Gene"].isin(gene_list)]
df_test_cell_lines_sample["Log10 Prot. count"] = df_test_cell_lines_sample["Log10 Prot. count"].replace(0,np.nan)
warnings.filterwarnings('ignore')
df_test_cell_lines = pd.DataFrame(columns= df_test_cell_lines_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_cell_lines_sample.copy()
    df_sample["Sample"] = i
    df_test_cell_lines = pd.concat([df_test_cell_lines, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set for each sample
df_test_cell_lines["Prot. count predict"] = np.nan
df_test_cell_lines = parallel_sim(df_test_cell_lines, df_lin, gene_list, num_samples, num_cores)
df_test_cell_lines.to_csv('results/rand_lin_model/linear_immune_test_cell_lines.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (cell lines)")

# Test models with data from different source (erythropoiesis data)
df_test_erythr = pd.read_csv("data/erythropoiesis_dataset.csv")
df_test_erythr_sample = df_test_erythr.loc[df_test_erythr["Gene"].isin(gene_list)]
df_test_erythr_sample["Log10 Prot. count"] = df_test_erythr_sample["Log10 Prot. count"].replace(0,np.nan)
warnings.filterwarnings('ignore')
df_test_erythr = pd.DataFrame(columns= df_test_erythr_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_erythr_sample.copy()
    df_sample["Sample"] = i
    df_test_erythr = pd.concat([df_test_erythr, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set for each sample
df_test_erythr["Prot. count predict"] = np.nan
df_test_erythr = parallel_sim(df_test_erythr, df_lin, gene_list, num_samples, num_cores)
df_test_erythr.to_csv('results/rand_lin_model/linear_immune_test_erythropoiesis.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (erythropoiesis)")

# Test models with data from different source (CCLE prot. abundance data)
df_test_cell_lines_abund = pd.read_csv('data/CCLE_prot_abundance_dataset2.tsv.gz', sep='\t')
df_test_cell_lines_abund_sample = df_test_cell_lines_abund.loc[df_test_cell_lines_abund["Gene"].isin(gene_list)]
df_test_cell_lines_abund_sample["Abundance"] = df_test_cell_lines_abund_sample["Abundance"].replace(0,np.nan)
warnings.filterwarnings('ignore')
df_test_cell_lines_abund = pd.DataFrame(columns= df_test_cell_lines_abund_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_cell_lines_abund_sample.copy()
    df_sample["Sample"] = i
    df_test_cell_lines_abund = pd.concat([df_test_cell_lines_abund, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set for each sample
df_test_cell_lines_abund["Prot. count predict"] = np.nan
df_test_cell_lines_abund = parallel_sim(df_test_cell_lines_abund, df_lin, gene_list, num_samples, num_cores)
df_test_cell_lines_abund.to_csv('results/rand_lin_model/linear_immune_test_CCLE.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (CCLE)")