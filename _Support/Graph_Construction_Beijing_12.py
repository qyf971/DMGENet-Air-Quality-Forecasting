import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import entropy
from scipy.stats import spearmanr
import numpy as np
import math
import os


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance (in kilometers) between two latitude and longitude coordinates.
    """
    # earth's radius in kilometers
    R = 6371.004

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def jensen_shannon_divergence(p, q):
    """
    Calculate the Jensen Shannon divergence of two probability distributions p and q.
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def jensen_shannon_distance(p, q):
    return math.sqrt(jensen_shannon_divergence(p, q))


def get_data(target):
    folder_path = './dataset/Beijing_12/AQI_processed/'
    file_names = [f'PRSA_Data_{i}.csv' for i in range(1, 13)]
    data_list = []
    for file_name in file_names:
        data = pd.read_csv(os.path.join(folder_path, file_name))
        if target == 'PM25':
            target_data = data['PM2.5'].values
        elif target == 'PM10':
            target_data = data['PM10'].values
        else:
            raise ValueError(f"Invalid target: {target}")
        data_list.append(target_data)
    return data_list


# distance graph
def calculate_the_distance_matrix(threshold):
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    stations = positions_data['site_name']
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')

    num_sites = len(stations)
    distance_matrix = np.zeros((num_sites, num_sites))

    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    std_deviation = np.std(distance_matrix.flatten())
    distance_matrix_exp = np.exp(-(distance_matrix ** 2) / (std_deviation ** 2))
    # np.fill_diagonal(distance_matrix_exp, 0.0)
    adj_matrix = np.where(distance_matrix_exp > threshold, distance_matrix_exp, 0.0)
    mask = distance_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = distance_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


def calculate_adjacency_matrix(R):
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')

    n = len(latitudes)
    A = np.zeros((n, n), dtype=float)
    edge_index = []
    edge_weights = []

    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if d_ij < R:
                    weight = 1 / d_ij
                    A[i, j] = weight
                    edge_index.append([i, j])
                    edge_weights.append(weight)

    edge_index = np.array(edge_index).T
    edge_weights = np.array(edge_weights)

    return A, edge_index, edge_weights


# neighbor graph
def calculate_the_neighbor_matrix():
    adj_matrix = pd.read_csv('./dataset/Beijing_12/neighbors/neighbors.csv', index_col=0).values.astype('float32')
    # np.fill_diagonal(adj_matrix, 0.0)
    mask = adj_matrix != 0
    edge_index = np.array(np.where(mask))
    return adj_matrix, edge_index


# similarity graph
def calculate_the_similarity_matrix(threshold, target):
    data_list = get_data(target)
    num_sites = len(data_list)
    js_divergence_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            js_divergence = jensen_shannon_distance(data_list[i], data_list[j])
            js_divergence_matrix[i, j] = js_divergence
            js_divergence_matrix[j, i] = js_divergence
    std_deviation = np.std(js_divergence_matrix.flatten())
    js_divergence_matrix_exp = np.exp(-(js_divergence_matrix ** 2) / (std_deviation ** 2))
    # np.fill_diagonal(js_divergence_matrix_exp, 0.0)
    adj_matrix = np.where(js_divergence_matrix_exp > threshold, js_divergence_matrix_exp, 0.0)
    mask = js_divergence_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = js_divergence_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


# correlation graph
def calculate_the_correlation_matrix(threshold, target):
    data_list = get_data(target)
    num_sites = len(data_list)
    correlation_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            if i ==j:
                correlation_matrix[i, j] = 1.0
            else:
                corr, _ = spearmanr(data_list[i], data_list[j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
    # np.fill_diagonal(correlation_matrix, 0.0)
    adj_matrix = np.where(correlation_matrix > threshold, correlation_matrix, 0.0)
    mask = correlation_matrix > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = correlation_matrix[mask]
    return adj_matrix, edge_index, edge_weights