import numpy as np
import math
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import shapely
import json
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from tqdm import tqdm

import scorers


def splits_to_regions_binary(row, splits):
    if splits is None:
        return 0
    if len(splits) == 0:
        return 0

    num = 0
    for v in splits:
        num = 2 * num + row[v]
    return num


def count_disjoints(geometry):
    if type(geometry) == Polygon:
        return 1
    else:
        return len(geometry.geoms)
    
def find_furtherst_angle(geo, pop_LON, pop_LAT):
    if type(geo) == shapely.geometry.polygon.Polygon:
        coords = np.array(geo.exterior.coords)
        relative_differences = coords - np.array([pop_LON, pop_LAT])
        angle = np.max(np.degrees(np.arctan2(relative_differences[:,0],relative_differences[:,1])))
    elif type(geo) == shapely.geometry.multipolygon.MultiPolygon:
        coords = np.concatenate([np.array(g.exterior.coords) for g in geo.geoms])
        relative_differences = coords - np.array([pop_LON, pop_LAT])
        angle = np.max(np.degrees(np.arctan2(relative_differences[:,0],relative_differences[:,1])))
    if angle < 0:
        angle += 360
    return angle


def split_this(
    df,
    n_districts,
    holder,
    n_split=1,
    pop_col="POPULATION",
    dem_col="G18GOVD",
    rep_col="G18GOVR",
    orig_dem=0.5,
    orig_rep=0.5,
    score_fn=scorers.abs_diff_parta_fn,
    n_sample_angles=None,
    degree_limit=0,
    dissolve_check=False,
    pivot_strategy="pcm",
):
    # print(n_districts)
    if n_districts != 1:
        # Get district ratios:
        if n_districts % 2 == 0:
            part_a = n_districts / 2
            part_b = n_districts / 2
        else:
            part_a = math.ceil(n_districts / 2)
            part_b = math.floor(n_districts / 2)

        # Get Total Population
        pop_total = df[pop_col].sum()

        if pivot_strategy == "pcm":
            print("pcm")
            # Get Population Center
            pop_center_LON = (df["RP_LON"] * df[pop_col]).sum() / pop_total
            pop_center_LAT = (df["RP_LAT"] * df[pop_col]).sum() / pop_total

            # Re-center LAT/LON relative to the Population Center
            df["RECENTERED_LON"] = df["RP_LON"] - pop_center_LON
            df["RECENTERED_LAT"] = df["RP_LAT"] - pop_center_LAT
        
        elif pivot_strategy == "centroid":
            # Default to Centroid
            centroid_LON, centroid_LAT = df.dissolve().centroid[0].coords[0]
            print("Centroid")

            # Re-center LAT/LON relative to the Centroid
            df["RECENTERED_LON"] = df["RP_LON"] - centroid_LON
            df["RECENTERED_LAT"] = df["RP_LAT"] - centroid_LAT
        else:
            raise ValueError("Only 'centroid' and 'pcm' are accepted pivot strategies at the moment.")



        # Get Angle by using ARCTAN(RECENTERED_LAT, RECENTERED_LON)
        df["RECENTERED_ANGLE"] = np.degrees(
            np.arctan2(df["RECENTERED_LAT"], df["RECENTERED_LON"])
        )
        df["RECENTERED_ANGLE"] = df["RECENTERED_ANGLE"].apply(
            lambda x: x if x > 0 else x + 360
        )

        # df["RECENTERED_ANGLE"] = df['geometry'].apply(find_furtherst_angle, pop_LON=pop_center_LON, pop_LAT=pop_center_LAT)

        ## LOOP for d in 0 to 359 ##
        start_angles = np.linspace(0, 359, 360)
        if n_sample_angles:
            start_angles = np.random.choice(start_angles, size=n_sample_angles)
        # Set Tracker for Angle Ranges and Competitiveness
        angle_ranges = []

        for d in tqdm(start_angles):
            # Reset Angle to be Relative to d
            df["RESETTED_ANGLE"] = df["RECENTERED_ANGLE"] - d
            df["RESETTED_ANGLE"] = df["RESETTED_ANGLE"].apply(
                lambda x: x if x > 0 else (x + 360)
            )
            df = df.sort_values("RESETTED_ANGLE").reset_index(drop=True)

            # Spin Clock Around till you achieve the desired ratio
            i = 0
            captured_pop = 0
            pop_threshold = (part_a) / (part_a + part_b) * pop_total
            while captured_pop < pop_threshold:
                captured_pop += df.loc[i, pop_col]
                i += 1
            final_angle = df.loc[i - 1, "RECENTERED_ANGLE"]

            ## Check if angle range loops past 360 for conditions:
            a1, a2 = d, final_angle

            # If specified, skip past any second hand angle that is not far away from the first handle angle.
            if abs(a2 - a1) <= degree_limit:
                continue

            ## Checking if swapping the last district will get us closer to the desired proportion.
            if a1 < a2:
                pop_check1 = df.loc[
                    (a1 <= df["RECENTERED_ANGLE"]) & (df["RECENTERED_ANGLE"] <= a2)
                ][
                    pop_col
                ].sum()  # right inclusive
                pop_check2 = df.loc[
                    (a1 <= df["RECENTERED_ANGLE"]) & (df["RECENTERED_ANGLE"] < a2)
                ][
                    pop_col
                ].sum()  # right exclusive
            else:
                pop_check1 = df.loc[
                    (df["RECENTERED_ANGLE"] >= a1) | (df["RECENTERED_ANGLE"] <= a2)
                ][
                    pop_col
                ].sum()  # right inclusive
                pop_check2 = df.loc[
                    (df["RECENTERED_ANGLE"] >= a1) | (df["RECENTERED_ANGLE"] < a2)
                ][
                    pop_col
                ].sum()  # right exclusive

            balance_1 = (pop_check1 / pop_total) * (1 - (pop_check1 / pop_total))
            balance_2 = (pop_check2 / pop_total) * (1 - (pop_check2 / pop_total))
            balance_desired = (
                (part_a) / (part_a + part_b) * (1 - (part_a) / (part_a + part_b))
            )

            inclusive = (
                True
                if abs(balance_1 - balance_desired) <= abs(balance_2 - balance_desired)
                else False
            )

            if a1 < a2:
                if inclusive:
                    df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                        lambda x: 1 if a1 <= x <= a2 else 0
                    )
                else:
                    df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                        lambda x: 1 if a1 <= x < a2 else 0
                    )
            elif a1 > a2:
                if inclusive:
                    df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                        lambda x: 1 if ((x >= a1) or (x <= a2)) else 0
                    )
                else:
                    df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                        lambda x: 1 if ((x >= a1) or (x < a2)) else 0
                    )

            # Dissolve Check for Non-Contiguity
            if dissolve_check:
                dissolved_df = df.dissolve(by=f"SPLIT_{n_split}")
                dissolved_df["disjoints"] = dissolved_df["geometry"].apply(
                    lambda x: count_disjoints(x)
                )
                dissolved_pieces = dissolved_df["disjoints"].sum()
            else:
                dissolved_pieces = 0

            df_a = df.loc[df[f"SPLIT_{n_split}"] == 1].copy().reset_index(drop=True)
            df_b = df.loc[df[f"SPLIT_{n_split}"] == 0].copy().reset_index(drop=True)

            # Metric Calculation, Change to any function chosen.
            score = score_fn(
                df_a, df_b, dem_col, rep_col, orig_dem=orig_dem, orig_rep=orig_rep
            )
            angle_ranges.append(
                {
                    "start": d,
                    "end": final_angle,
                    "score": score,
                    "disjoints": dissolved_pieces,
                    "inclusive": inclusive
                }
            )

        ars = pd.DataFrame(angle_ranges)
        # ars['entropy_prod'] = (ars['entropy_a']*ars["entropy_b"])/(ars['entropy_a']+ars["entropy_b"])
        # ars = ars.sort_values("entropy_prod", ascending=False).reset_index(drop=True)
        ars = ars.sort_values(["disjoints", "score"], ascending=True).reset_index(
            drop=True
        )

        a1, a2 = ars.iloc[0]["start"], ars.iloc[0]["end"]
        inclusive = ars.iloc[0]["inclusive"]
        ## Check if angle range loops past 360 for conditions:
        if a1 < a2:
            if inclusive:
                df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                    lambda x: 1 if a1 <= x <= a2 else 0
                )
            else:
                df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                    lambda x: 1 if a1 <= x < a2 else 0
                )
        elif a1 > a2:
            if inclusive:
                df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                    lambda x: 1 if ((x >= a1) or (x <= a2)) else 0
                )
            else:
                df[f"SPLIT_{n_split}"] = df["RECENTERED_ANGLE"].apply(
                    lambda x: 1 if ((x >= a1) or (x < a2)) else 0
                )

        df_a = df.loc[df[f"SPLIT_{n_split}"] == 1].copy().reset_index(drop=True)
        df_b = df.loc[df[f"SPLIT_{n_split}"] == 0].copy().reset_index(drop=True)

        split_this(
            df_a,
            part_a,
            holder=holder,
            n_split=n_split + 1,
            pop_col=pop_col,
            dem_col=dem_col,
            rep_col=rep_col,
            orig_dem=orig_dem,
            orig_rep=orig_rep,
            n_sample_angles=n_sample_angles,
            degree_limit=degree_limit,
            score_fn=score_fn,
            pivot_strategy=pivot_strategy,
        )
        split_this(
            df_b,
            part_b,
            holder=holder,
            n_split=n_split + 1,
            pop_col=pop_col,
            dem_col=dem_col,
            rep_col=rep_col,
            orig_dem=orig_dem,
            orig_rep=orig_rep,
            n_sample_angles=n_sample_angles,
            degree_limit=degree_limit,
            score_fn=score_fn,
            pivot_strategy=pivot_strategy,
        )

    else:
        holder.append(df)


def split_review(
    input_df,
    split,
    split_cols,
    pop,
    d_votes,
    r_votes,
    plot_party=False,
    figsize=(8, 6),
    save=None,
):
    input_df["TEMP_DISTRICT"] = input_df.apply(
        splits_to_regions_binary, splits=split_cols[:split], axis=1
    )
    n_districts = input_df["TEMP_DISTRICT"].nunique()
    # random_districts = np.random.choice(np.linspace(0,n_districts-1,n_districts), size=n_districts, replace=False, p=None)
    random_districts = np.linspace(0, n_districts - 1, n_districts)
    shuffle_dict = dict(zip(input_df["TEMP_DISTRICT"].unique(), random_districts))
    input_df["TEMP_DISTRICT"] = input_df["TEMP_DISTRICT"].apply(
        lambda x: shuffle_dict[x]
    )

    districts_df = input_df.dissolve(
        by="TEMP_DISTRICT",
        aggfunc={"TEMP_DISTRICT": "first", d_votes: "sum", r_votes: "sum"},
    )
    districts_df["Dem_Ratio"] = districts_df[d_votes] / (
        districts_df[d_votes] + districts_df[r_votes]
    )
    districts_df["Rep_Ratio"] = districts_df[r_votes] / (
        districts_df[d_votes] + districts_df[r_votes]
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if plot_party:
        base = districts_df.plot(
            ax=ax,
            cmap="RdBu",
            column="Dem_Ratio",
            legend=True,
            vmin=0,
            vmax=1,
            edgecolor="black",
        )
        dem_wins = districts_df.loc[
            districts_df["Dem_Ratio"] >= districts_df["Rep_Ratio"]
        ].shape[0]
        rep_wins = districts_df.loc[
            districts_df["Dem_Ratio"] < districts_df["Rep_Ratio"]
        ].shape[0]
        title = f"Districts: {districts_df.shape[0]} | D-R: {dem_wins}-{rep_wins}"
    else:
        base = districts_df.plot(ax=ax, cmap="tab20b", edgecolor="black")
        title = f"Districts - {districts_df.shape[0]}"

    points = []
    for x in range(2**split):
        sub = input_df.loc[input_df["TEMP_DISTRICT"] == x]
        if sub.shape[0] == 0:
            continue
        pop_center_LON = (sub["RP_LON"] * sub[pop]).sum() / sub[pop].sum()
        pop_center_LAT = (sub["RP_LAT"] * sub[pop]).sum() / sub[pop].sum()
        point = shapely.geometry.Point(pop_center_LON, pop_center_LAT)
        points.append(point)
    pop_center_df = gpd.GeoDataFrame(geometry=points)
    pop_center_df.plot(ax=ax, color="white", linewidth=1.0, edgecolor="black")
    plt.title(title)
    if save:
        fig.savefig(save)
    plt.show()

    return districts_df

def folium_mapper(df, map_type):
    pass