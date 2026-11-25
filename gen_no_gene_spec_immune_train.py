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

num_samples = 100
num_cores = 24

# Get train/test splits 
df_train = pd.read_csv("results/lin_model/linear_immune_train.tsv.gz",sep="\t")
df_train = df_train[['Dataset type','RNA dataset num.','Prot. count dataset num.','Cell','Gene','Log2 TPM','Log10 Prot. count','Sample']]
df_test = pd.read_csv("results/lin_model/linear_immune_validation.tsv.gz",sep="\t")
df_test = df_test[['Dataset type','RNA dataset num.','Prot. count dataset num.','Cell','Gene','Log2 TPM','Log10 Prot. count','Sample']]
gene_list = list(set(df_train["Gene"].drop_duplicates().to_list()+df_test["Gene"].drop_duplicates().to_list()))
print("Number of genes studied: "+str(len(gene_list)))
print("Data sampling done!")

# Generate linear model for all samples
df_lin_sample = pd.DataFrame(columns=["Sample","Intercept","Slope","r2_train","MAE_train"])
for sample in df_train["Sample"].drop_duplicates():
    df_train_sample = df_train.loc[df_train["Sample"]==sample]
    TPM = df_train_sample["Log2 TPM"].values
    Prot = df_train_sample["Log10 Prot. count"].values
    model = TheilSenRegressor(random_state=0).fit(TPM.reshape(-1, 1), Prot)
    df_lin_sample.loc[len(df_lin_sample.index)] = [sample, model.intercept_, model.coef_[0], r2_score(Prot, model.predict(TPM.reshape(-1, 1))),np.mean(abs(model.predict(TPM.reshape(-1, 1))-Prot))]
df_lin_sample.to_csv('results/lin_model/no_gene_specific/linear_immune.tsv.gz', sep='\t', index=False, compression='gzip')
print("Linear models done!")

# Simulate for the training set of each sample
df_train["Prot. count predict"] = np.nan
df_train["Log10 Prot. count"] = df_train["Log10 Prot. count"].replace(0,np.nan)
for sample in df_train["Sample"].drop_duplicates():
    df_train.loc[df_train["Sample"]==sample,"Prot. count predict"] = df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Intercept"].values[0]+df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Slope"].values[0]*df_train.loc[df_train["Sample"]==sample,"Log2 TPM"]
df_train.to_csv('results/lin_model/no_gene_specific/linear_immune_train.tsv.gz', sep='\t', index=False, compression='gzip')
print("Training set simulation done!")

# Simulate for the validation set of each sample
df_test["Prot. count predict"] = np.nan
df_test["Log10 Prot. count"] = df_test["Log10 Prot. count"].replace(0,np.nan)
for sample in df_test["Sample"].drop_duplicates():
    df_test.loc[df_test["Sample"]==sample,"Prot. count predict"] = df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Intercept"].values[0]+df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Slope"].values[0]*df_test.loc[df_test["Sample"]==sample,"Log2 TPM"]
df_test.to_csv('results/lin_model/no_gene_specific/linear_immune_validation.tsv.gz', sep='\t', index=False, compression='gzip')
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

# Simulate for the training set of each sample
df_test_tissues["Prot. count predict"] = np.nan
for sample in df_test_tissues["Sample"].drop_duplicates():
    df_test_tissues.loc[df_test_tissues["Sample"]==sample,"Prot. count predict"] = df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Intercept"].values[0]+df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Slope"].values[0]*df_test_tissues.loc[df_test_tissues["Sample"]==sample,"Log2 TPM"]
df_test_tissues.to_csv('results/lin_model/no_gene_specific/linear_immune_test_tissues.tsv.gz', sep='\t', index=False, compression='gzip')
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

# Simulate the test set 2 for each sample
df_test_cell_lines["Prot. count predict"] = np.nan
for sample in df_test_cell_lines["Sample"].drop_duplicates():
    df_test_cell_lines.loc[df_test_cell_lines["Sample"]==sample,"Prot. count predict"] = df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Intercept"].values[0]+df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Slope"].values[0]*df_test_cell_lines.loc[df_test_cell_lines["Sample"]==sample,"Log2 TPM"]
df_test_cell_lines.to_csv('results/lin_model/no_gene_specific/linear_immune_test_cell_lines.tsv.gz', sep='\t', index=False, compression='gzip')
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

# Simulate the test set 2 for each sample
df_test_erythr["Prot. count predict"] = np.nan
for sample in df_test_erythr["Sample"].drop_duplicates():
    df_test_erythr.loc[df_test_erythr["Sample"]==sample,"Prot. count predict"] = df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Intercept"].values[0]+df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Slope"].values[0]*df_test_erythr.loc[df_test_erythr["Sample"]==sample,"Log2 TPM"]
df_test_erythr.to_csv('results/lin_model/no_gene_specific/linear_immune_test_erythropoiesis.tsv.gz', sep='\t', index=False, compression='gzip')
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
for sample in df_test_cell_lines_abund["Sample"].drop_duplicates():
    df_test_cell_lines_abund.loc[df_test_cell_lines_abund["Sample"]==sample,"Prot. count predict"] = df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Intercept"].values[0]+df_lin_sample.loc[df_lin_sample["Sample"]==sample,"Slope"].values[0]*df_test_cell_lines_abund.loc[df_test_cell_lines_abund["Sample"]==sample,"Log2 TPM"]
df_test_cell_lines_abund.to_csv('results/lin_model/no_gene_specific/linear_immune_test_CCLE.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (CCLE)")