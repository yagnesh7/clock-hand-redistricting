from multiprocessing import Pool, cpu_count
import numpy as np
import geopandas as gpd
import pandas as pd
import time
import os


def rotate_vector(orig_vector, degree, origin=np.array([0, 0])):
    """
    Rotate a 2-D vector (point) by some given degree. Optionally around a specified origin besides (0,0).
    Supports multiple points.

    Parameters
    ----------
    orig_vector : np.array
                 Original vector(s) of size (2,) or (n,2)
    degree : float
             Counter-clockwise rotation amount in degrees
    origin : np.array
            Origin of rotation
    """
    vector = np.array(orig_vector)
    origin = np.array(origin)

    if vector.ndim == 1:
        vector = np.expand_dims(vector, axis=0)
    vector = vector - origin  # center of circle

    theta = np.radians(degree)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    vector = (rotation_matrix @ vector.T).T
    vector = vector + origin
    return vector


def reock_score(geom):
    geom_area = geom.geometry.area[0]
    min_circle = geom.geometry.minimum_bounding_circle()[0]
    min_circle_area = min_circle.area
    reock_score = geom_area / min_circle_area
    return reock_score


def get_slice(degree, df, origin, points, target_pop, pop, n_districts, step):
    df = df.sort_values(by="INDEX_").reset_index(drop=True)
    df["NEW_DISTRICT_"] = 0

    rotated_points = rotate_vector(points, 360 - degree, origin)
    df["ROTATED_LON"] = rotated_points[:, 0]
    df = df.sort_values(by="ROTATED_LON").reset_index(drop=True)
    i = 0
    captured_pop = 0
    while captured_pop < target_pop:
        captured_pop += df.loc[i, pop]
        i += 1

    # Check for Inclusivity
    pop_check1 = df.iloc[: i - 1][pop].sum()
    pop_check2 = df.iloc[:i][pop].sum()
    inclusive = (
        True if abs(pop_check1 - target_pop) > abs(pop_check2 - target_pop) else False
    )
    if inclusive:
        df.loc[: i - 1, "NEW_DISTRICT_"] = 1
    else:
        df.loc[: i - 2, "NEW_DISTRICT_"] = 1

    df_a = df.loc[df["NEW_DISTRICT_"] == 1].copy().reset_index(drop=True)
    df_b = df.loc[df["NEW_DISTRICT_"] != 1].copy().reset_index(drop=True)

    weight = 1# 1/(n_districts + 1 - step)
    df_a_compactness = reock_score(df_a.dissolve())
    df_b_compactness = 0 #reock_score(df_b.dissolve())
    mean_compactness = (weight*df_a_compactness) + ((1-weight)*df_b_compactness) #(df_a_compactness + df_b_compactness) / 2

    return {
        "degree": degree,
        "pieces": i,
        "inclusive": inclusive,
        "population": captured_pop,
        "compactness": mean_compactness,
    }


def prepare_data(df_, d_votes="USH20_D", r_votes="USH20_R"):
    df = df_.copy().reset_index(drop=True)
    df[d_votes] = df[d_votes].fillna(0)
    df[r_votes] = df[r_votes].fillna(0)
    df["REPRESENTATIVE_POINT"] = df.centroid
    df["RP_LON"] = df["REPRESENTATIVE_POINT"].apply(lambda p: p.x)
    df["RP_LAT"] = df["REPRESENTATIVE_POINT"].apply(lambda p: p.y)

    df["ORIGINAL_INDEX_"] = df.index
    df["DISTRICT_"] = 0  # Original
    return df


if __name__ == "__main__":
    data_path = "../data/mggg-states/"
    output_dir = "../output/"
    run_name = "moving_knife_c"
    seed = 42

    pop = "POP20"
    d_votes = "USH20_D"
    r_votes = "USH20_R"


    state_dict = dict(
        az_2020={"districts": 9, "crs": "EPSG:2223"},
        ca_2020={"districts": 52, "crs": "EPSG:2225"},
        ct_2020={"districts": 5, "crs": "EPSG:2234"},
        fl_2020={"districts": 28, "crs": "EPSG:2225"},
        ga_2020={"districts": 14, "crs": "EPSG:2240"},
        ma_2020={"districts": 9, "crs": "EPSG:2249"},
        md_2020={"districts": 8, "crs": "EPSG:2248"},
        mi_2020={"districts": 13, "crs": "EPSG:2252"},
        mn_2020={"districts": 8, "crs": "EPSG:2811"},
        nc_2020={"districts": 14, "crs": "EPSG:2264"},
        nh_2020={"districts": 2, "crs": "EPSG:3437"},
        nj_2020={"districts": 12, "crs": "EPSG:2824"},
        nv_2020={"districts": 4, "crs": "EPSG:2821"},
        oh_2020={"districts": 15, "crs": "EPSG:2834"},
        or_2020={"districts": 6, "crs": "EPSG:2269"},
        pa_2020={"districts": 17, "crs": "EPSG:2271"},
        sc_2020={"districts": 7, "crs": "EPSG:2273"},
        tx_2020={"districts": 38, "crs": "EPSG:2277"},
        va_2020={"districts": 11, "crs": "EPSG:2283"},
        wi_2020={"districts": 8, "crs": "EPSG:2288"},
    )

    for i, s in enumerate(state_dict.keys()):
        state_file = s #s.split("\\")[-1]
        n_districts = state_dict[state_file]["districts"]
        print(i, state_file, n_districts)

        voting_precincts = gpd.read_file(f"../redistricting/data_prep/output/{state_file}/")
        orig_columns = list(voting_precincts.columns)
        df = prepare_data(voting_precincts)
        target_pop_ratio = 1 / n_districts
        target_pop = target_pop_ratio * df[pop].sum()

        degrees = np.array(range(0, 360, 10), dtype="f")


        holder = []
        step = 1
        while step < n_districts:
            start = time.time()
            # df will get smaller as algorithm loops (pieces get allocated)
            df["INDEX_"] = df.index
            points = df[["RP_LON", "RP_LAT"]].to_numpy()

            # Create circle of remaining pieces
            state_dissolved = df.dissolve()
            state_min_radius = state_dissolved.geometry.minimum_bounding_radius()[0]
            state_min_circle = state_dissolved.geometry.minimum_bounding_circle()[0]
            state_min_circle_o = state_min_circle.centroid
            origin = np.array([state_min_circle_o.x, state_min_circle_o.y])

            # Degree Loop
            with Pool(cpu_count()) as p:
                degrees_tracker = p.starmap(
                    get_slice, [(d, df, origin, points, target_pop, pop, n_districts, step) for d in degrees]
                )
            degrees_tracker = pd.DataFrame(degrees_tracker)
            degrees_tracker = degrees_tracker.sort_values(
                by="compactness", ascending=False
            ).reset_index(drop=True)
            optimal_choice = degrees_tracker.iloc[0]

            df = df.sort_values(by="INDEX_").reset_index(drop=True)
            rotated_points = rotate_vector(points, 360 - optimal_choice["degree"], origin)
            df["ROTATED_LON"] = rotated_points[:, 0]
            df = df.sort_values(by="ROTATED_LON").reset_index(drop=True)
            if optimal_choice["inclusive"]:
                df.loc[: optimal_choice["pieces"] - 1, "DISTRICT_"] = step
            else:
                df.loc[: optimal_choice["pieces"] - 2, "DISTRICT_"] = step

            holder.append(df.loc[df["DISTRICT_"] == step])
            print('step:',step,  '- districts:', df.shape[0], '-', round(time.time() - start, 2),'seconds')
            df = df.loc[df["DISTRICT_"] == 0]  # So far unused.
  
            step += 1
            
        # Final
        df["DISTRICT_"] = step
        holder.append(df)

        all_districts = pd.concat(holder).reset_index(drop=True)
        # print(all_districts)
        # all_districts.plot(column="DISTRICT_")
        os.makedirs(os.path.join(output_dir, state_file), exist_ok=True)
        all_districts[orig_columns + ["DISTRICT_"]].to_file(os.path.join(output_dir, state_file,f"{state_file}_{run_name}.shp"))
