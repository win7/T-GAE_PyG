import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg

from tqdm import tqdm

# New for preprocessing MS data
def get_subgroups_id(df_join_raw, groups):
    dict_groups_id = {}
    for group in groups:
        # get group
        columns = list(df_join_raw.filter(like=group).columns)

        subgroups = [item.split("{}_".format(group))[1].split(".")[0] for item in columns]
        subgroups = np.unique(subgroups)
        dict_groups_id[group] = subgroups.tolist()
    return dict_groups_id

def check_dataset(df):
    print("Checking dataset") 
    
    # ds = df.isin([np.inf, -np.inf]) 
    # print(ds.head()) 
    
    count_inf = np.isinf(df).values.sum() 
    print("Count infinite:\t", count_inf) 
    
    # printing column name where infinity is present 
    """ print() 
    print("Column name where infinity is present") 
    col_name = df.columns.to_series()[np.isinf(df).any()] 
    print(col_name) """
    
    count_nan = df.isna().sum().sum()
    print("Count nan:\t", count_nan)
    
    count_negative = (df < 0).sum().sum()
    print("Count negative:\t", count_negative) 
    
    count_zero = (df == 0).sum().sum()
    print("Count zero:\t", count_zero) 
    
    count_positive = (df > 0).sum().sum()
    print("Count positive:\t", count_positive)
    
def log10_global(df_join_raw):
    df_join_raw_log = df_join_raw.copy()
    for column in df_join_raw.columns:
        # df_join_raw_log[column] = np.log10(df_join_raw[column], where=df_join_raw[column]>0)
        df_join_raw_log[column] = np.log10(df_join_raw_log[column])
        df_join_raw_log[column] = df_join_raw_log[column].replace(-np.Inf, np.nan)
        df_join_raw_log[column] = df_join_raw_log[column].replace(np.nan, df_join_raw_log[column].min() / 100)
    return df_join_raw_log

def split_groups_subgroups(df_join_raw_log, groups_id, subgroups_id):
    dict_df_groups_subgroups = {}
    for group_id in groups_id:
        df_aux = df_join_raw_log.filter(like=group_id)
        dict_aux = {}
        
        for subgroup_id in subgroups_id[group_id]:
            dict_aux[subgroup_id] = df_aux.filter(like="{}_{}.".format(group_id, subgroup_id))
        dict_df_groups_subgroups[group_id] = dict_aux
    return dict_df_groups_subgroups

def transpose(df):
    df = df.T
    df.reset_index(drop=True, inplace=True)
    return df

def transpose_global(dict_groups_subgroups):
    dict_groups_subgroups_t = {}
    for group_id, dict_groups in dict_groups_subgroups.items():
        dict_aux = {}
        for subgroup_id, df_subgroup in dict_groups.items():
            aux_t = transpose(df_subgroup)
            dict_aux[subgroup_id] = aux_t
        dict_groups_subgroups_t[group_id] = dict_aux
    return dict_groups_subgroups_t

def correlation_global(exp, dict_groups_subgroups_t):
    dict_groups_subgroups_t_corr = {}
    for group_id, dict_groups in dict_groups_subgroups_t.items():
        dict_aux = {}
        for subgroup_id, df_subgroup in dict_groups.items():
            print(group_id, subgroup_id, df_subgroup.shape)
            
            """ import numpy as np
            cov = np.cov(df_subgroup.values, rowvar=False)
            cond = np.linalg.cond(cov)
            print("Condition number:", cond, cond > 1e8) # ill-conditioned if > 1e8 (True, instable) """

            matrix = pg.pcorr(df_subgroup)
            dict_aux[subgroup_id] = matrix
            
            matrix.to_csv("experiments/output/{}/correlations/{}_{}.csv".format(exp, group_id, subgroup_id), index=True)
        dict_groups_subgroups_t_corr[group_id] = dict_aux
    return dict_groups_subgroups_t_corr

def build_graph_weight_global_directed(exp, dict_groups_subgroups_t_corr, threshold=0.3):
    dict_groups_subgroups_t_corr_g = {}
    for group_id, dict_groups in dict_groups_subgroups_t_corr.items():
        dict_aux = {}
        for subgroup_id, df_subgroup in dict_groups.items():
            df_weighted_edges = (df_subgroup.where(np.triu(np.ones(df_subgroup.shape), k=1).astype(bool)).stack())
            df_weighted_edges = df_weighted_edges.dropna().to_frame()
            df_weighted_edges.reset_index(inplace=True)
            df_weighted_edges.columns = ["source", "target", "weight"]
            df_weighted_edges = df_weighted_edges[df_weighted_edges["weight"].abs() >= threshold]
            df_weighted_edges["subgroup"] = [subgroup_id] * len(df_weighted_edges)
            dict_aux[subgroup_id] = df_weighted_edges
            
            df_weighted_edges.to_csv("experiments/output/{}/preprocessing/edges/{}_{}.csv".format(exp, group_id, subgroup_id), index=False)
            # G = nx.from_pandas_edgelist(df_weighted_edges, "source", "target", edge_attr=["weight"])
            # print(groups_id[i], subgroups_id[groups_id[i]][j], G.number_of_nodes(), G.number_of_edges())
            # nx.write_gexf(G, "experiments/output/{}/preprocessing/graphs/graphs_{}_{}.gexf".format(exp, groups_id[i], subgroups_id[groups_id[i]][j]))
        dict_groups_subgroups_t_corr_g[group_id] = dict_aux
    return dict_groups_subgroups_t_corr_g