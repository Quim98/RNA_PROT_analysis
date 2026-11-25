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

# Get transcriptomics and proteomics data from dataset not used for training (indep. test)
df = pd.read_csv("data/whole_dataset.csv").dropna()
# Get data from only 1 RNA and protein dataset in immune cells
df = df.loc[(df["RNA dataset num."]==1)&(df["Prot. count dataset num."]==1)]
df = df.loc[~df["Cell"].isin(["T4.CM_IL2RAlow","T4.EM_IL2RAlow","T4.naive_IL2RAlow"])]

# Only get genes with enough data and that are differentially expressed
Num_data = [len(df.loc[df["Gene"]==gene].index) for gene in df["Gene"].drop_duplicates()]
var_TPM = [df.loc[df["Gene"]==gene,"Log2 TPM"].var() for gene in df["Gene"].drop_duplicates()[Num_data >= np.median(Num_data)]]
gene_list1 = df["Gene"].drop_duplicates()[Num_data >= np.median(Num_data)][var_TPM >= np.median(var_TPM)].values

# Get transcriptomics and proteomics data from dataset USED FOR TRAINING
df = pd.read_csv("data/tissues_dataset.csv").dropna()

# Only get genes with enough data and that are differentially expressed
Num_data = [len(df.loc[df["Gene"]==gene].index) for gene in df["Gene"].drop_duplicates()]
var_TPM = [df.loc[df["Gene"]==gene,"Log2 TPM"].var() for gene in df["Gene"].drop_duplicates()[Num_data >= np.median(Num_data)]]
gene_list2 = df["Gene"].drop_duplicates()[Num_data >= np.median(Num_data)][var_TPM >= np.median(var_TPM)].values

# Get intersection of genes with enough data and that are differentially expressed
gene_list = np.intersect1d(gene_list1, gene_list2)
print("Number of genes studied: "+str(len(gene_list)))
df = df.loc[df["Gene"].isin(gene_list)]

# Separate data into training and test 10 times (80/20 split)
df_train = pd.DataFrame(columns=list(df.columns)+["Sample"])
df_test = pd.DataFrame(columns=list(df.columns)+["Sample"])
warnings.filterwarnings('ignore')
for i in range(0,num_samples):
    df_train_slice, df_test_slice = train_test_split(df, test_size=0.2, random_state=i)
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
df_lin.to_csv('results/lin_model/linear_tissues.tsv.gz', sep='\t', index=False, compression='gzip')
print("Linear models done!")

# Simulate for the training set of each sample
df_train["Prot. count predict"] = np.nan
df_train = parallel_sim(df_train, df_lin, gene_list, num_samples, num_cores)
# Get metrics for training set
df_train_res = pd.DataFrame(columns=["Sample","r2","MAE","Spearman (TPMxProt)","Spearman (ModelxProt)"])
for i in range(0,num_samples):
    df_train_res.loc[len(df_train_res)] = [i,r2_score(df_train.loc[df_train["Sample"]==i,"Log10 Prot. count"], df_train.loc[df_train["Sample"]==i,"Prot. count predict"]),(df_train.loc[df_train["Sample"]==i,"Log10 Prot. count"]-df_train.loc[df_train["Sample"]==i,"Prot. count predict"]).abs().mean(),spearmanr(df_train.loc[df_train["Sample"]==i,"Log2 TPM"],df_train.loc[df_train["Sample"]==i,"Log10 Prot. count"]).statistic,spearmanr(df_train.loc[df_train["Sample"]==i,"Prot. count predict"],df_train.loc[df_train["Sample"]==i,"Log10 Prot. count"]).statistic]
df_train.to_csv('results/lin_model/linear_tissues_train.tsv.gz', sep='\t', index=False, compression='gzip')
df_train_res.to_csv('results/lin_model/linear_tissues_train_metrics.tsv.gz', sep='\t', index=False, compression='gzip')
print("Training set simulation done!")
    
# Simulate for the test set of each sample
df_test["Prot. count predict"] = np.nan
df_test = parallel_sim(df_test, df_lin, gene_list, num_samples, num_cores)
# Get metrics for test set
df_test_res = pd.DataFrame(columns=["Sample","r2","MAE","Spearman (TPMxProt)","Spearman (ModelxProt)"])
for i in range(0,num_samples):
    df_test_res.loc[len(df_test_res)] = [i,r2_score(df_test.loc[df_test["Sample"]==i,"Log10 Prot. count"], df_test.loc[df_test["Sample"]==i,"Prot. count predict"]),(df_test.loc[df_test["Sample"]==i,"Log10 Prot. count"]-df_test.loc[df_test["Sample"]==i,"Prot. count predict"]).abs().mean(),spearmanr(df_test.loc[df_test["Sample"]==i,"Log2 TPM"],df_test.loc[df_test["Sample"]==i,"Log10 Prot. count"]).statistic,spearmanr(df_test.loc[df_test["Sample"]==i,"Prot. count predict"],df_test.loc[df_test["Sample"]==i,"Log10 Prot. count"]).statistic]
df_test.to_csv('results/lin_model/linear_tissues_test.tsv.gz', sep='\t', index=False, compression='gzip')
df_test_res.to_csv('results/lin_model/linear_tissues_test_metrics.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done!")
    
# Test models with data from different source (immune data)
df_test_immune = pd.read_csv("data/whole_dataset.csv").dropna()
df_test_immune = df_test_immune.loc[(df_test_immune["RNA dataset num."]==1)&(df_test_immune["Prot. count dataset num."]==1)]
df_test_immune = df_test_immune.loc[~df_test_immune["Cell"].isin(["T4.CM_IL2RAlow","T4.EM_IL2RAlow","T4.naive_IL2RAlow"])]
df_test_immune_sample = df_test_immune.loc[df_test_immune["Gene"].isin(gene_list)]
warnings.filterwarnings('ignore')
df_test_immune = pd.DataFrame(columns= df_test_immune_sample.columns.to_list()+["Sample"])
for i in range(0,num_samples):
    df_sample = df_test_immune_sample.copy()
    df_sample["Sample"] = i
    df_test_immune = pd.concat([df_test_immune, df_sample], ignore_index=True)
warnings.filterwarnings('default')

# Simulate the test set 2 for each sample
df_test_immune["Prot. count predict"] = np.nan
df_test_immune = parallel_sim(df_test_immune, df_lin, gene_list, num_samples, num_cores)

df_test_immune_res = pd.DataFrame(columns=["Sample","r2","MAE","Spearman (TPMxProt)","Spearman (ModelxProt)"])
for i in range(0,num_samples):
    df_test_immune_res.loc[len(df_test_immune_res)] = [i,r2_score(df_test_immune.loc[df_test_immune["Sample"]==i,"Log10 Prot. count"], df_test_immune.loc[df_test_immune["Sample"]==i,"Prot. count predict"]),(df_test_immune.loc[df_test_immune["Sample"]==i,"Log10 Prot. count"]-df_test_immune.loc[df_test_immune["Sample"]==i,"Prot. count predict"]).abs().mean(),spearmanr(df_test_immune.loc[df_test_immune["Sample"]==i,"Log2 TPM"],df_test_immune.loc[df_test_immune["Sample"]==i,"Log10 Prot. count"]).statistic,spearmanr(df_test_immune.loc[df_test_immune["Sample"]==i,"Prot. count predict"],df_test_immune.loc[df_test_immune["Sample"]==i,"Log10 Prot. count"]).statistic]
df_test_immune.to_csv('results/lin_model/linear_tissues_test_immune.tsv.gz', sep='\t', index=False, compression='gzip')
df_test_immune_res.to_csv('results/lin_model/linear_tissues_test_immune_metrics.tsv.gz', sep='\t', index=False, compression='gzip')
print("Test set simulation done! (immune)")