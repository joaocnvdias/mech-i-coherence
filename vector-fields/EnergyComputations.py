import torch

def compute_layer_vectors(layer_activation):
    return layer_activation[1:]-layer_activation[:-1]

def compute_vectors(hidden_states):
    return [compute_layer_vectors(layer) for layer in hidden_states]

def compute_dot_product_layer(layer_vectors):
    return torch.stack([torch.dot(layer_vectors[i,:],layer_vectors[i+1,:]) for i in range(layer_vectors.shape[0]-1)])

def compute_dot_product(vector_transitions_trajectory):
    return [compute_dot_product_layer(layer_vectors) for layer_vectors in vector_transitions_trajectory]  

def average_layer_dot_product(layer_dot_product):
    return layer_dot_product.mean()

def average_dot_product(dot_product_list):
    return torch.stack([average_layer_dot_product(layer_dot_product) for layer_dot_product in dot_product_list])

def sum_layer_energy(average_layer_dot_product):
    return average_layer_dot_product.sum()

def energy_pipeline(layer_hidden_states):
    if not isinstance(layer_hidden_states, list):
        raise TypeError("Expected a list of tensors (one per layer + embedding layer).")
    return sum_layer_energy(average_dot_product(compute_dot_product(compute_vectors(layer_hidden_states)))).item()

