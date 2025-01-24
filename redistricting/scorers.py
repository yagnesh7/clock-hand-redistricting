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
    dem_prop_1 = summed_pops_1[dem_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    rep_prop_1 = summed_pops_1[rep_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    score = entropy([dem_prop_1, rep_prop_1], base=2)

    # Will always minimize, for entropy negate
    return -score


def entropy_score_mean_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    rep_prop_1 = summed_pops_1[rep_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    score_1 = entropy([dem_prop_1, rep_prop_1], base=2)

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )
    rep_prop_2 = summed_pops_2[rep_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )
    score_2 = entropy([dem_prop_2, rep_prop_2], base=2)

    score = (score_1 + score_2) / 2  # Average of both
    # Will always minimize, for entropy negate
    return -score


def abs_diff_parta_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    rep_prop_1 = summed_pops_1[rep_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    score = abs(dem_prop_1 - rep_prop_1)
    return score


def abs_diff_mean_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes

    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    rep_prop_1 = summed_pops_1[rep_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    score_1 = abs(dem_prop_1 - rep_prop_1)

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )
    rep_prop_2 = summed_pops_2[rep_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )
    score_2 = abs(dem_prop_2 - rep_prop_2)

    score = score_1 + score_2
    return score


def match_statewide_mean_fn(df_1, df_2, dem_col, rep_col, **kwargs):
    orig_dem = kwargs["orig_dem"]
    orig_rep = kwargs["orig_rep"]
    # Always takes in both dataframes
    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    rep_prop_1 = summed_pops_1[rep_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )
    score_1 = abs(dem_prop_1 - orig_dem) + abs(rep_prop_1 - orig_rep)

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )
    rep_prop_2 = summed_pops_2[rep_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )
    score_2 = abs(dem_prop_2 - orig_dem) + abs(rep_prop_2 - orig_rep)

    score = (score_1 + score_2) / 2  # Average of both
    return score


def equalize_voting(df_1, df_2, dem_col, rep_col, **kwargs):
    # Always takes in both dataframes
    # Calculate values
    summed_pops_1 = df_1[[dem_col, rep_col]].sum()
    dem_prop_1 = summed_pops_1[dem_col] / (
        summed_pops_1[rep_col] + summed_pops_1[dem_col]
    )

    summed_pops_2 = df_2[[dem_col, rep_col]].sum()
    dem_prop_2 = summed_pops_2[dem_col] / (
        summed_pops_2[rep_col] + summed_pops_2[dem_col]
    )

    score = abs(dem_prop_1 - dem_prop_2)
    return score


def schwartzberg_score(geom):
    perimeter = geom.length
    area = geom.area

    numerator = 4 * math.pi * area
    denominator = perimeter**2
    score = math.sqrt(numerator / denominator)

    return score


def convex_hull_score(geom):
    area = geom.area[0]
    convex_hull_area = geom.convex_hull.area[0]

    score = area / convex_hull_area
    return score


def polsby_popper_score(geom):
    perimeter = geom.length[0]
    area = geom.area[0]

    numerator = 4 * math.pi * area
    denominator = perimeter**2
    score = numerator / denominator

    return score


def reock_score(geom):
    geom_area = geom.geometry.area[0]
    min_circle = geom.geometry.minimum_bounding_circle()[0]
    min_circle_area = min_circle.area
    reock_score = geom_area / min_circle_area
    return reock_score

def length_width_score(geom):
        bounds = geom.bounds

        # Generate points
        minx = bounds.loc[0]["minx"]
        maxx = bounds.loc[0]["maxx"]
        miny = bounds.loc[0]["miny"]
        maxy = bounds.loc[0]["maxy"]

        side1 = maxx - minx
        side2 = maxy - miny
        if side1 > side2:
            numerator = side2
            denominator = side1
        else:
            numerator = side1
            denominator = side2
        score = numerator/denominator
        return score

def min_perimeter_score(geom):
    perimeter = geom.length[0]
    return -perimeter