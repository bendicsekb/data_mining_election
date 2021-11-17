from math import log10, sqrt
import pandas as pd
import numpy as np

import data_refinement as refine

# function which routes to the specified quality measure
def set_quality(description: refine.Description, data: refine.DataSet, method: refine.Method):
    if method == refine.Method.OUR:
        return our_quality_measure(description, data, "EUCLIDEAN")
    elif method in [refine.Method.NORM, refine.Method.LABELWISE, refine.Method.PAIRWISE]:
        return wouters_quality_measure(description=description, data=data, method=method)
    else:
        raise Exception("Quality measure not defined:", method)


# function which computes our currently defined quality measure
def our_quality_measure(description: refine.Description, data: refine.DataSet, distance_function: str):
    subgroup_data = refine.get_subgroup_data(description, data.dataframe)  # get subgroup rows from data_refinement
    complement_data = data.dataframe.drop(labels=subgroup_data.index, axis="rows")  # obtain the complement rows
    subgroup_rows = len(subgroup_data.index)
    complement_rows = len(complement_data.index)
    all_rows = len(data.dataframe.index)

    if complement_rows == 0:
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
    subgroup_size_factor = {
        "original": ((subgroup_rows * (subgroup_rows - 1)) / complement_rows),
        # -(n/N)lg(n/N)-(nC/N)lg(nC/N)
        "entropy-based": -((subgroup_rows/all_rows)*log10(subgroup_rows/all_rows))-((complement_rows/all_rows)*log10(complement_rows/all_rows))
    }
    description.quality = (numerator / (denominator + 1)) * subgroup_size_factor["entropy-based"]


# compute distance between input vector and matrix given the specified distance function
def compute_distance(vector: [int], matrix, function: str):
    if function == "EUCLIDEAN":
        return np.linalg.norm(vector - matrix)
    else:
        raise Exception("Distance function", function, "has no implemented function")

# Wouter Duivesteijn paper
def wouters_quality_measure(description: refine.Description, data: refine.DataSet, method: refine.Method):
    target_data = refine.get_subgroup_data(description, data.dataframe)[data.targets].to_numpy()
    # Calculate M<pi> (preference matrix representing the subgroup)
    Mpis = np.zeros((len(target_data), *2*[len(data.targets)]))
    for index in range(len(target_data)):
        targets = target_data[index]
        for ii in range(len(targets)):
            for jj in range(len(targets)):
                lambda_i = targets[ii]
                lambda_j = targets[jj]
                Mpis[index, ii,jj] = data.omega(lambda_i, lambda_j)
    # Ld = MD - MS -> the distance matrix
    MS = 1 / len(target_data) * np.sum(Mpis, axis=0)
    Ld = data.MD - MS

    # sqrt(s/n) The sqrt of the fraction of the dataset covered by s: Size<s>
    normalization_factor = sqrt(len(target_data)/len(data.dataframe))
    quality = 0
    # Calculate quality based on method 
    if method == refine.Method.NORM:
        quality = normalization_factor * np.linalg.norm(Ld)
    elif method == refine.Method.LABELWISE:
        row_sums = np.sum(Ld, axis= 1)
        max_row = np.max(row_sums)
        quality = normalization_factor * 1/(Ld.shape[0]-1) * max_row
    elif method == refine.Method.PAIRWISE:
        max_elem = Ld.max()
        quality = normalization_factor * max_elem
    description.quality = quality

