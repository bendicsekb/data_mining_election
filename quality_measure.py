import pandas as pd
import numpy as np

import data_refinement as refine


# function which routes to the specified quality measure
def set_quality(description: refine.Description, data: refine.DataSet, quality_measure_id: int, logging_dataframe: pd.DataFrame):
    if quality_measure_id == 0:
        return our_quality_measure(description, data, "EUCLIDEAN", logging_dataframe)
    else:
        raise Exception("Quality measure not defined:", quality_measure_id)


# function which computes our currently defined quality measure
def our_quality_measure(description: refine.Description, data: refine.DataSet, distance_function: str, logging_dataframe: pd.DataFrame):
    subgroup_data = refine.get_subgroup_data(description, data.dataframe)  # get subgroup rows from data_refinement
    complement_data = pd.concat([data.dataframe, subgroup_data]).drop_duplicates(keep=False)  # obtain the complement rows
    subgroup_rows = len(subgroup_data.index)
    complement_rows = len(complement_data.index)

    if subgroup_rows == 0 or complement_rows <= 1:
        description.quality = 0.0
        return

    d_x_x_sum = 0
    d_x_y_sum = 0
    sub_target_matrix = subgroup_data[data.targets]
    comp_target_matrix = complement_data[data.targets]
    for sub_index, sub_row in subgroup_data.iterrows():
        sub_target_vector = sub_row[data.targets]
        d_x_x_sum += compute_distance(sub_target_vector, sub_target_matrix, distance_function)
        d_x_y_sum += compute_distance(sub_target_vector, comp_target_matrix, distance_function)

    numerator = d_x_y_sum  # (1 / (subgroup_rows * complement_rows)) * d_x_y_sum
    denominator = d_x_x_sum  # (1 / (complement_rows * (complement_rows - 1))) * d_x_x_sum

    description.quality = (numerator / (denominator + 1)) * ((subgroup_rows * (subgroup_rows - 1)) / complement_rows)

    logged_info = pd.DataFrame(data=[[description.to_string(), round(description.quality, 4), subgroup_rows, complement_rows, round(numerator, 4), round(denominator, 4)]], columns=["Description", "Quality", "Size of subgroup", "Size of complement", "Numerator", "Denominator"])
    return logging_dataframe.append(logged_info)


# compute distance between input vector and matrix given the specified distance function
def compute_distance(vector, matrix, function: str):
    if function is "EUCLIDEAN":
        return np.linalg.norm(vector - matrix)
    else:
        raise Exception("Distance function", function, "has no implemented function")

