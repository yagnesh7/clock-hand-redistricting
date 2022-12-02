import numpy as np
import math
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy 

from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

def entropy_score_parta_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    rep_prop_1 = summed_pops_1[rep_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    score = entropy([dem_prop_1, rep_prop_1], base=2)

    # Will always minimize, for entropy negate
    return -score

def entropy_score_mean_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    rep_prop_1 = summed_pops_1[rep_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    score_1 = entropy([dem_prop_1, rep_prop_1], base=2)

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])
    rep_prop_2 = summed_pops_2[rep_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])
    score_2 = entropy([dem_prop_2, rep_prop_2], base=2)

    score = (score_1+score_2)/2 # Average of both
    # Will always minimize, for entropy negate
    return -score
    

def abs_diff_parta_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    rep_prop_1 = summed_pops_1[rep_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    score = abs(dem_prop_1-rep_prop_1) 
    return score
    
def abs_diff_mean_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    rep_prop_1 = summed_pops_1[rep_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    score_1 = abs(dem_prop_1-rep_prop_1) 

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])
    rep_prop_2 = summed_pops_2[rep_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])
    score_2 = abs(dem_prop_2-rep_prop_2)

    score = (score_1+score_2) # Average of both
    return score

def match_statewide_mean_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    orig_dem = kwargs["orig_dem"]
    orig_rep = kwargs["orig_rep"]
    # Always takes in both dataframes
    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    rep_prop_1 = summed_pops_1[rep_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])
    score_1 = abs(dem_prop_1-orig_dem)+abs(rep_prop_1-orig_rep)

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])
    rep_prop_2 = summed_pops_2[rep_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])
    score_2 = abs(dem_prop_2-orig_dem)+abs(rep_prop_2-orig_rep)

    score = (score_1+score_2)/2 # Average of both
    return score


def equalize_voting(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes
    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col]/(summed_pops_1[rep_col]+summed_pops_1[dem_col])

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col]/(summed_pops_2[rep_col]+summed_pops_2[dem_col])

    score = abs(dem_prop_1-dem_prop_2) 
    return score