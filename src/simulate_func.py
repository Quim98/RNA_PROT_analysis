import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TheilSenRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import warnings
from multiprocessing import Pool
from joblib import Parallel,delayed,parallel_config,dump,load

def lin_model_sample(list_df):
    df_sample_data = list_df[0]
    df_sample_lin = list_df[1]
    gene_list = list_df[2]
    for gene in gene_list:
        df_sample_data.loc[df_sample_data["Gene"]==gene,"Prot. count predict"] = df_sample_lin.loc[df_sample_lin["Gene"]==gene,"Intercept"].values[0]+df_sample_lin.loc[df_sample_lin["Gene"]==gene,"Slope"].values[0]*df_sample_data.loc[df_sample_data["Gene"]==gene,"Log2 TPM"]
    return df_sample_data

def parallel_sim(df_data, df_lin, gene_list, num_samples, num_cores):
    param_parallel = []
    for i in range(0,num_samples):
        param_parallel.append([df_data.loc[df_data["Sample"]==i],df_lin.loc[df_lin["Sample"]==i],gene_list])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(lin_model_sample, param_parallel)
    return pd.concat(out_vec, axis=0)

def gen_lin_model(list_df):
    df_sample = list_df[0]
    gene_list = list_df[1]
    df_lin_sample = pd.DataFrame(columns=["Sample","Gene","Intercept","Slope","r2_train","MAE_train"])
    for gene in gene_list:
        TPM = df_sample.loc[df_sample["Gene"]==gene,"Log2 TPM"].values
        Prot = df_sample.loc[df_sample["Gene"]==gene,"Log10 Prot. count"].values
        model = TheilSenRegressor(random_state=0).fit(TPM.reshape(-1, 1), Prot)
        df_lin_sample.loc[len(df_lin_sample.index)] = [df_sample["Sample"].values[0], gene, model.intercept_, model.coef_[0], r2_score(Prot, model.predict(TPM.reshape(-1, 1))),np.mean(abs(model.predict(TPM.reshape(-1, 1))-Prot))]
    return df_lin_sample

def parallel_gen_lin_model(df_data, gene_list, num_samples, num_cores):
    param_parallel = []
    for i in range(0,num_samples):
        param_parallel.append([df_data.loc[df_data["Sample"]==i],gene_list])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(gen_lin_model, param_parallel)
    return pd.concat(out_vec, axis=0)

def gen_lasso_model(list_df):
    df_TPM_sample = list_df[0]
    TPM_mat = df_TPM_sample[df_TPM_sample.columns[:-2]].values
    df_prot_sample = list_df[1]
    gene_list = list_df[2]
    alfa_val = list_df[3]
    dict_models = {}
    df_models_sample = pd.DataFrame(columns=["Sample","Gene","r2_train","MAE_train"])
    for gene in gene_list:
        Prot = df_prot_sample[gene].values
        model = Lasso(alpha=alfa_val)  # Higher alpha->More generalization, Lower alpha->More overfit 
        model.fit(TPM_mat, Prot)
        dict_models[gene] = model
        df_models_sample.loc[len(df_models_sample.index)] = [df_TPM_sample["Sample"].values[0], gene, r2_score(Prot, model.predict(TPM_mat)),np.mean(abs(Prot-model.predict(TPM_mat)))]
    return dict_models, df_models_sample

def parallel_gen_lasso_model(df_data_TPM, df_data_prot, gene_list, num_samples, num_cores, alfa_val):
    param_parallel = []
    for i in range(0,num_samples):
        param_parallel.append([df_data_TPM.loc[df_data_TPM["Sample"]==i], df_data_prot.loc[df_data_TPM["Sample"]==i], gene_list, alfa_val])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(gen_lasso_model, param_parallel)
    return [i[0] for i in out_vec],pd.concat([i[1] for i in out_vec], axis=0)

def lasso_model_sample(list_df):
    df_TPM_sample = list_df[0]
    df_res_sample = list_df[1]
    filename_models =  list_df[2]
    sample = list_df[3]
    list_models = load(filename_models)
    model_dict = list_models[sample]
    del(list_models)
    for gene in df_res_sample["Gene"].drop_duplicates().values:
        df_TPM_sample_gene = df_TPM_sample.set_index('Cell').reindex(df_res_sample.loc[df_res_sample["Gene"]==gene,"Cell"])
        df_res_sample.loc[df_res_sample["Gene"]==gene,"Prot. count predict"] = model_dict[gene].predict(df_TPM_sample_gene[df_TPM_sample_gene.columns[:-1]].values)
    return df_res_sample

def parallel_sim_lasso(df_data_TPM, df_melted, filename_models, num_samples, num_cores):
    param_parallel = []
    for i in range(0, num_samples):
        param_parallel.append([df_data_TPM.loc[df_data_TPM["Sample"] == i], df_melted.loc[df_melted["Sample"] == i], filename_models, i])
    # Use joblib's Parallel with delayed function for parallel execution (multiprocessing pool could only be used once per script)
    with parallel_config(backend='threading', n_jobs=num_cores):
        out_vec = Parallel()(delayed(lasso_model_sample)(params) for params in param_parallel)
    return pd.concat(out_vec, axis=0)

def parallel_sim_lasso_all_data(df_data_TPM, df_melted, filename_models, num_samples, num_cores):
    param_parallel = []
    for i in range(0, num_samples):
        df_melted["Sample"] = i
        param_parallel.append([df_data_TPM.loc[df_data_TPM["Sample"] == i], df_melted.copy(), filename_models, i])
    # Use joblib's Parallel with delayed function for parallel execution (multiprocessing pool could only be used once per script)
    with parallel_config(backend='threading', n_jobs=num_cores):
        out_vec = Parallel()(delayed(lasso_model_sample)(params) for params in param_parallel)
    return pd.concat(out_vec, axis=0)