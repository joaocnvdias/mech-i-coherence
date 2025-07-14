import torch
from math import acos

def compute_layer_vectors(layer_activation):
    return layer_activation[1:]-layer_activation[:-1] #matrix except first_row - matrix except last_row

def compute_vectors(hidden_states):
    return [compute_layer_vectors(layer) for layer in hidden_states]

def compute_angle_layer(layer_vectors):
    angles=[]

    for i in range(layer_vectors.shape[0]-1):
        a = layer_vectors[i,:]
        b = layer_vectors[i+1,:]

        angles.append(acos(torch.dot(a, b) / (torch.norm(a) * torch.norm(b))))

    return torch.tensor(angles, dtype=torch.bfloat16)

def compute_angle(vector_transitions_trajectory):
    return [compute_angle_layer(layer_vectors) for layer_vectors in vector_transitions_trajectory]

def average_layer_angle(layer_dot_product):
    # return layer_dot_product.nanmean()
    return layer_dot_product.nanmean()

def average_angle(dot_product_list):
    return torch.stack([average_layer_angle(layer_dot_product) for layer_dot_product in dot_product_list])

def sum_layer_energy(average_layer_dot_product):
    return average_layer_dot_product.sum()

def energy_pipeline_angles(layer_hidden_states):
    if not isinstance(layer_hidden_states, list):
        raise TypeError("Expected a list of tensors (one per layer + embedding layer).")
    return sum_layer_energy(average_angle(compute_angle(compute_vectors(layer_hidden_states)))).item()