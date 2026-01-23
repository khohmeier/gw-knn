import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import ot
import pandas as pd
import time
from sklearn.metrics import f1_score
import random


"""
Graph generators & graph properties
"""
def add_random_edges(graph, num_nodes, max_edges, max_weight=10):
  """Adds a random number of edges to a graph with random weights.
  Args:
    graph: The graph to add edges to.
    num_nodes: The number of nodes in the graph.
    max_edges: The maximum number of edges to add.
    max_weight: The maximum weight of an edge. By default set to 10.
  Returns:
    None.
    """
  for _ in range(random.randint(1, max_edges)):
    # Randomly select two different nodes and a random weight for the edge
    u, v = random.sample(range(num_nodes), 2)
    weight = random.randint(1, max_weight)
    graph.add_edge(u, v, weight=weight)


def generate_erdos_renyi_graphs(n_graphs, n_nodes, p_edge):
  """
  Generates a list of Erdos-Renyi graphs of given size and probability of edge addition.
  
  Args:
    n_graphs (int): Number of graphs to generate
    n_nodes (int): Number of nodes in each graph
    p_edge (float): Probability of edge addition

  Returns:
    graphs: the list of generated graphs with a label
  """
  graphs = []
  for _ in range(n_graphs):
    graph = nx.erdos_renyi_graph(n_nodes, p_edge)
    graphs.append((graph, 1))  # Label as 1
  return graphs


def generate_sbm_graphs(n_graphs, block_sizes, prob_matrix):
  """
  Generates a list of Stochastic Block Model graphs of given number of nodes, block sizes, and probabilities of edge addition within blocks and between blocks.
  Args:
    n_graphs (int): Number of graphs to generate
    block_sizes (list): List of block sizes for each graph
    prob_matrix (list): List of probabilities of edge addition within blocks and between blocks for each graph

  Returns:
    graphs: the list of generated graphs with a label
  """
  graphs = []
  for _ in range(n_graphs):
    graph = nx.stochastic_block_model(block_sizes, prob_matrix)
    graphs.append((graph, -1))  # Label as -1
  return graphs


def compute_distance_matrix(graph):
  """Computes the pairwise distance matrix for a graph. Takes into account weights on edges.
  Args:
      graph: A NetworkX graph.
  Returns:
      A numpy array representing the pairwise distance matrix where the value at
      (i, j) is the shortest path length between node i and node j. If i and j are
      not connected, the distance will be set to 0 instead of inf.
  """
  # Initialize a matrix of zeros (instead of infinities)
  num_nodes = len(graph.nodes)
  distance_matrix = np.full((num_nodes, num_nodes), 10000.0)
  # Compute shortest paths and update the distance matrix
  for node in graph.nodes():
    lengths = nx.shortest_path_length(graph, source=node, weight="weight")
    for target, length in lengths.items():
      distance_matrix[node, target] = length
  return distance_matrix


"""
Gromov-Wasserstein barycenter visualizations
"""
def weight_thresholding(graph1, graph2, threshold, adjacency_threshold, rotangle=0, mapping=None, facecolor="white", nodecolor="lightblue", edgecolor="grey", fontcolor="red"):
    """
    mapping: a dictionary indicating relabeling of nodes. By default, set to None; otherwise, pass in a dictionary of nodes in the form original:new.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax in axes:
      ax.set_facecolor(facecolor)
    # Visualize graph1
    axes[0].set_title('Graph 1')
    pos = nx.shell_layout(graph1)
    nx.draw_networkx_edge_labels(graph1,
                                 pos,
                                 edge_labels={
                                     (u, v): d['weight']
                                     for u, v, d in graph1.edges(data=True)
                                 },
                                 ax=axes[0])
    nx.draw(graph1,
            pos,
            ax=axes[0],
            with_labels=True,
            node_color=nodecolor,
            edge_color=edgecolor)
    # Iterate over weights_list to compute and visualize barycenters
    weights_list = [[2 / 3, 1 / 3], [0.5, 0.5], [1 / 3, 2 / 3]]
    for index, weights in enumerate(weights_list):
      barycenter_matrix = compute_gromov_wasserstein_barycenter(
          graph1, graph2, weights, num_nodes_barycenter)
      # Convert the barycenter's distance matrix to an adjacency matrix using the threshold
      barycenter_adjacency = threshold_distance_matrix(barycenter_matrix,
                                                       adjacency_threshold)
      # Compute edge weights/exp for visualization
      barycenter_edges_weight = np.exp(-barycenter_matrix) * barycenter_adjacency
      # Convert adjacency matrix to graph
      barycenter_graph = nx.from_numpy_array(barycenter_adjacency,
                                             create_using=nx.Graph)
      if mapping is not None:
          barycenter_graph = nx.relabel_nodes(barycenter_graph,
                                          mapping)  # Apply relabeling
      nx.set_edge_attributes(barycenter_graph, {
          (i, j): barycenter_edges_weight[i, j]
          for i, j in barycenter_graph.edges()
      }, 'weight')
      edges_to_remove = [(u, v) for u, v, d in barycenter_graph.edges(data=True)
                         if d['weight'] < threshold]
      barycenter_graph.remove_edges_from(edges_to_remove)
      # Scale edges based on weights
      edges, weights = zip(
          *nx.get_edge_attributes(barycenter_graph, 'weight').items())
      # Visualize with edge thickness corresponding to weight
      bary_pos = nx.shell_layout(barycenter_graph)
      rotated_pos = rotate_positions(bary_pos, rotangle)
      axes[index + 1].set_title(f'Barycenter {index + 1}')
      widths = np.array([max(w * 15.0, 0.1) for w in weights], dtype=float)
      nx.draw_networkx_edge_labels(
          barycenter_graph,
          rotated_pos,
          edge_labels={
              (u, v): f"{d['weight']:.5f}"
              for u, v, d in barycenter_graph.edges(data=True)
              if (u, v) not in edges_to_remove
          },
          font_color=fontcolor,
          ax=axes[index + 1])
      nx.draw(barycenter_graph,
              rotated_pos,
              ax=axes[index + 1],
              with_labels=True,
              node_color=nodecolor,
              edge_color=edgecolor,
              width=widths)
    # Visualize graph2
    axes[4].set_title('Graph 2')
    pos = nx.shell_layout(graph2)
    nx.draw_networkx_edge_labels(graph2,
                                 pos,
                                 edge_labels={
                                     (u, v): d['weight']
                                     for u, v, d in graph2.edges(data=True)
                                 },
                                 ax=axes[4])
    nx.draw(graph2,
            pos,
            ax=axes[4],
            with_labels=True,
            node_color=nodecolor,
            edge_color=edgecolor)
    plt.tight_layout()
    #plt.savefig('barycenter_graphs_trimmed_and_rotated_relabeled.png')
    plt.show()


def compute_gromov_wasserstein_barycenter(graph1, graph2, weights,
                                          num_nodes_barycenter):
  # Compute distance matrices for both graphs
  distance_matrix1 = compute_distance_matrix(graph1)
  distance_matrix2 = compute_distance_matrix(graph2)
  # Prepare the list of distance matrices and weights
  distance_matrices = [distance_matrix1, distance_matrix2]
  # Compute the barycenter's distance matrix
  barycenter_matrix = ot.gromov.gromov_barycenters(N=num_nodes_barycenter,
                                                   Cs=distance_matrices,
                                                   lambdas=weights,
                                                   loss_fun='square_loss',
                                                   random_state=30)
  return barycenter_matrix
    

def threshold_distance_matrix(distance_matrix, upper_threshold):
  adjacency_matrix = np.copy(distance_matrix)
  # Set distances greater than or equal to the threshold and initially set distances (indicating no connection) to 0
  adjacency_matrix[(adjacency_matrix >= upper_threshold) |
                   (adjacency_matrix == 10000.0)] = 0
  # Set all other positive distances to 1, indicating an edge exists
  adjacency_matrix[(adjacency_matrix < upper_threshold)
                   & (adjacency_matrix > 0)] = 1
  return adjacency_matrix


def rotate_positions(pos, angle_degree):
    angle_rad = np.radians(angle_degree)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_pos = {}
    for node, original_pos in pos.items():
        rotated_pos[node] = np.dot(rotation_matrix, original_pos)
    return rotated_pos


"""
GW-NCC and fGW-NCC classification algorithms
"""
def gw_barycentersubset_class_alg(distance_matrices,
                                  labels,
                                  num_folds,
                                  num_nodes,
                                  method,
                                  seed=42,
                                  log=False):
    """Implements GW-NCC binary classification algorithm with the GW barycenter of each class being computed with a 20% subset of the training data.
    Parameters:
    distance_matrices: list of pairwise distance matrices
    labels: list of labels for each graph
    num_folds: number of folds for cross-validation
    num_nodes: number of nodes for the GW barycenter of each class
    method: cross-validation method, either "kfold" (for k-fold CV) or "stratk" (for stratified k-fold CV)
    seed: random seed for reproducibility
    log: boolean flag to print additional information
    
    Returns:
    class_acc_list: list of classification accuracies for each fold
    negone_acc_list: list of classification accuracies for class -1 for each fold
    one_acc_list: list of classification accuracies for class 1 for each fold
    f1_score_list: list of F1 scores for each fold
    negone_barys: list of GW barycenters for class -1 for each fold
    one_barys: list of GW barycenters for class 1 for each fold"""
    class_acc_list = [
    ]  # list to store classification accuracies from each fold
    negone_acc_list = []  # list to store class -1 accuracies from each fold
    one_acc_list = []  # list to store class 1 accuracies from each fold
    f1_score_list = []  # list to store F1 scores from each fold
    negone_barys = []  # list to store class -1 barycenters from each fold
    one_barys = []  # list to store class 1 barycenters from each fold

    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(k_folds.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log is True:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
        X_train, X_test = [distance_matrices[idx] for idx in train_index
                           ], [distance_matrices[idx] for idx in test_index]
        y_train, y_test = [labels[idx] for idx in train_index
                           ], [labels[idx] for idx in test_index]

        X_train_negone = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        X_train_one = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == 1
        ]

        # Randomly select 20% of each category
        subset_negone = random.sample(X_train_negone,
                                      k=max(1,
                                            len(X_train_negone) // 5))
        subset_one = random.sample(X_train_one,
                                      k=max(1,
                                            len(X_train_one) // 5))

        # Compute the GW barycenter for each category using the subset
        barycenter_matrix_negone = ot.gromov.gromov_barycenters(
            N=num_nodes,
            Cs=subset_negone,
            loss_fun='square_loss',
            random_state=seed)
        if log is True:
            print("GW barycenter for Class -1 graphs:",
                  barycenter_matrix_negone, f"Fold {i}")
        negone_barys.append(barycenter_matrix_negone)
        print("GW barycenter for Class -1 graphs computed")

        barycenter_matrix_one = ot.gromov.gromov_barycenters(
            N=num_nodes,
            Cs=subset_one,
            loss_fun='square_loss',
            random_state=seed)
        one_barys.append(barycenter_matrix_one)
        if log is True:
            print("GW barycenter for Class 1 graphs:",
                  barycenter_matrix_one, f"Fold {i}")
        print("GW barycenter for Class 1 graphs computed")
        gw, log_dists = ot.gromov.gromov_wasserstein(barycenter_matrix_negone,
                                                     barycenter_matrix_one,
                                                     loss_fun='square_loss',
                                                     log=True)
        print("Distance between barycenters: ", log_dists['gw_dist'],
              f" Fold {i}")

        correct_count = 0
        correct_count_negone = 0
        correct_count_one = 0
        total_count_negone = 0
        total_count_one = 0
        predicted_labels = []
        true_labels = []
        total_samples = len(X_test)
        for idx, graph in enumerate(X_test):
            gw_negone, log_negone = ot.gromov.gromov_wasserstein(
                graph,
                barycenter_matrix_negone,
                loss_fun='square_loss',
                log=True)
            gw_one, log_one = ot.gromov.gromov_wasserstein(
                graph,
                barycenter_matrix_one,
                loss_fun='square_loss',
                log=True)
            ground_truth_label = y_test[idx]
            true_labels.append(ground_truth_label)

            # Check if classification is correct
            predicted_label = -1 if log_negone['gw_dist'] < log_one[
                'gw_dist'] else 1
            predicted_labels.append(predicted_label)

            if ground_truth_label == -1:
                total_count_negone += 1
                if predicted_label == ground_truth_label:
                    correct_count_negone += 1
            if ground_truth_label == 1:
                total_count_one += 1
                if predicted_label == ground_truth_label:
                    correct_count_one += 1
            if predicted_label == ground_truth_label:
                correct_count += 1

        # Calculate classification accuracy
        classification_accuracy = (correct_count / total_samples) * 100
        negone_accuracy = (correct_count_negone / total_count_negone) * 100
        one_accuracy = (correct_count_one / total_count_one) * 100

        # Calculate F1 scores
        average_type = "macro" if method == "kfold" else "weighted"
        f1_weighted = f1_score(
            true_labels, predicted_labels, average=average_type) * 100

        print(f"F1 Score ({average_type} avg): {f1_weighted}%, Fold {i}")
        print(f"Classification Accuracy: {classification_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class -1: {negone_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class 1: {one_accuracy}%",
              f"Fold {i}")
        class_acc_list.append(classification_accuracy)
        negone_acc_list.append(negone_accuracy)
        one_acc_list.append(one_accuracy)
        f1_score_list.append(f1_weighted)

    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)

    # Compute and print averages and standard deviations
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    avg_classification_acc = np.mean(class_acc_list)
    std_classification_acc = np.std(class_acc_list)
    print(
        f"Overall Classification Accuracy: {avg_classification_acc:.2f}% ± {std_classification_acc:.2f}%"
    )

    avg_negone_acc = np.mean(negone_acc_list)
    std_negone_acc = np.std(negone_acc_list)
    print(
        f"Class -1 Classification Accuracy: {avg_negone_acc:.2f}% ± {std_negone_acc:.2f}%"
    )

    avg_one_acc = np.mean(one_acc_list)
    std_one_acc = np.std(one_acc_list)
    print(
        f"Class 1 Classification Accuracy: {avg_one_acc:.2f}% ± {std_one_acc:.2f}%"
    )

    avg_f1_score = np.mean(f1_score_list)
    std_f1_score = np.std(f1_score_list)
    average_type = "macro" if method == "kfold" else "weighted"
    print(
        f"F1 Score ({average_type} avg): {avg_f1_score:.2f}% ± {std_f1_score:.2f}%"
    )

    return class_acc_list, negone_acc_list, one_acc_list, f1_score_list, negone_barys, one_barys


def gw_barycenter_class_alg_var_nodes(distance_matrices,
                                      labels,
                                      num_folds,
                                      num_nodes_negone,
                                      num_nodes_one,
                                      method,
                                      seed=42,
                                      log=False):
    """Implements GW-NCC binary classification algorithm with variable number of nodes for the GW barycenter of each class.
    Parameters:
    distance_matrices: list of pairwise distance matrices
    labels: list of labels for each graph
    num_folds: number of folds for cross-validation
    num_nodes_negone: number of nodes for the GW barycenter of class -1
    num_nodes_one: number of nodes for the GW barycenter of class 1
    method: cross-validation method, either "kfold" (for k-fold 
    CV) or "stratk" (for stratified k-fold CV)
    seed: random seed for reproducibility
    log: boolean flag to print additional information
    
    Returns:
    class_acc_list: list of classification accuracies for each fold
    negone_acc_list: list of classification accuracies for class -1 for each fold
    one_acc_list: list of classification accuracies for class 1 for each fold
    f1_score_list: list of F1 scores for each fold
    negone_barys: list of GW barycenters for class -1 for each fold
    one_barys: list of GW barycenters for class 1 for each fold"""
    
    class_acc_list = [
    ]  # list to store classification accuracies from each fold
    negone_acc_list = []  # list to store class -1 accuracies from each fold
    one_acc_list = []  # list to store class 1 accuracies from each fold
    f1_score_list = []  # list to store F1 scores from each fold
    negone_barys = []  # list to store class -1 barycenters from each fold
    one_barys = []  # list to store class 1 barycenters from each fold

    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(k_folds.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log is True:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
        X_train, X_test = [distance_matrices[idx] for idx in train_index
                           ], [distance_matrices[idx] for idx in test_index]
        y_train, y_test = [labels[idx] for idx in train_index
                           ], [labels[idx] for idx in test_index]
        X_train_negone = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        X_train_one = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == 1
        ]
        # Compute the GW barycenter for each category
        barycenter_matrix_negone = ot.gromov.gromov_barycenters(
            N=num_nodes_negone,
            Cs=X_train_negone,
            loss_fun='square_loss',
            random_state=seed)
        if log is True:
            print("GW barycenter for Class -1 graphs:",
                  barycenter_matrix_negone, f"Fold {i}")
        negone_barys.append(barycenter_matrix_negone)
        print("GW barycenter for Class -1 graphs computed")
        barycenter_matrix_one = ot.gromov.gromov_barycenters(
            N=num_nodes_one,
            Cs=X_train_one,
            loss_fun='square_loss',
            random_state=seed)
        one_barys.append(barycenter_matrix_one)
        if log is True:
            print("GW barycenter for Class 1graphs:",
                  barycenter_matrix_one, f"Fold {i}")
        print("GW barycenter for Class 1 graphs computed")

        gw, log_dists = ot.gromov.gromov_wasserstein(barycenter_matrix_negone,
                                                     barycenter_matrix_one,
                                                     loss_fun='square_loss',
                                                     log=True)
        print("Distance between barycenters: ", log_dists['gw_dist'],
              f" Fold {i}")

        correct_count = 0
        correct_count_negone = 0
        correct_count_one = 0
        total_count_negone = 0
        total_count_one = 0

        predicted_labels = []
        true_labels = []
        total_samples = len(X_test)
        for idx, graph in enumerate(X_test):
            gw_negone, log_negone = ot.gromov.gromov_wasserstein(
                graph,
                barycenter_matrix_negone,
                loss_fun='square_loss',
                log=True)
            gw_one, log_one = ot.gromov.gromov_wasserstein(
                graph,
                barycenter_matrix_one,
                loss_fun='square_loss',
                log=True)
            ground_truth_label = y_test[idx]
            true_labels.append(ground_truth_label)
            # Check if classification is correct
            predicted_label = -1 if log_negone['gw_dist'] < log_one[
                'gw_dist'] else 1
            predicted_labels.append(predicted_label)
            if ground_truth_label == -1:
                total_count_negone += 1
                if predicted_label == ground_truth_label:
                    correct_count_negone += 1
            if ground_truth_label == 1:
                total_count_one += 1
                if predicted_label == ground_truth_label:
                    correct_count_one += 1
            if predicted_label == ground_truth_label:
                correct_count += 1
        # Calculate classification accuracy
        classification_accuracy = (correct_count / total_samples) * 100
        negone_accuracy = (correct_count_negone / total_count_negone) * 100
        one_accuracy = (correct_count_one / total_count_one) * 100

        # Calculate F1 scores
        average_type = "macro" if method == "kfold" else "weighted"
        f1_weighted = f1_score(
            true_labels, predicted_labels, average=average_type) * 100

        print(f"F1 Score ({average_type} avg): {f1_weighted}%, Fold {i}")
        print(f"Classification Accuracy: {classification_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class -1: {negone_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class 1: {one_accuracy}%",
              f"Fold {i}")
        class_acc_list.append(classification_accuracy)
        negone_acc_list.append(negone_accuracy)
        one_acc_list.append(one_accuracy)
        f1_score_list.append(f1_weighted)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)

    # Compute and print averages and standard deviations
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    avg_classification_acc = np.mean(class_acc_list)
    std_classification_acc = np.std(class_acc_list)
    print(
        f"Overall Classification Accuracy: {avg_classification_acc:.2f}% ± {std_classification_acc:.2f}%"
    )

    avg_negone_acc = np.mean(negone_acc_list)
    std_negone_acc = np.std(negone_acc_list)
    print(
        f"Class -1 Classification Accuracy: {avg_negone_acc:.2f}% ± {std_negone_acc:.2f}%"
    )

    avg_one_acc = np.mean(one_acc_list)
    std_one_acc = np.std(one_acc_list)
    print(
        f"Class 1 Classification Accuracy: {avg_one_acc:.2f}% ± {std_one_acc:.2f}%"
    )

    avg_f1_score = np.mean(f1_score_list)
    std_f1_score = np.std(f1_score_list)
    average_type = "macro" if method == "kfold" else "weighted"
    print(
        f"F1 Score ({average_type} avg): {avg_f1_score:.2f}% ± {std_f1_score:.2f}%"
    )

    return class_acc_list, negone_acc_list, one_acc_list, f1_score_list, negone_barys, one_barys


### Fused Gromov-Wasserstein NCC ###
def fgw_barycenter_class_alg_var_nodes(distance_matrices,
                                       graphs_list,
                                       labels,
                                       num_folds,
                                       num_nodes_negone,
                                       num_nodes_one,
                                       method,
                                       seed=42,
                                       log=False):
    """Implements FGW-NCC binary classification algorithm with variable number of nodes for the FGW barycenter of each class.
    Parameters:
    distance_matrices: list of pairwise distance matrices
    graphs_list: list of graphs
    labels: list of labels for each graph
    num_folds: number of folds for cross-validation
    num_nodes_negone: number of nodes for the FGW barycenter of class -1
    num_nodes_one: number of nodes for the FGW barycenter of class 1
    method: cross-validation method, either "kfold" (for k-fold
    CV) or "stratk" (for stratified k-fold CV)
    seed: random seed for reproducibility
    log: boolean flag to print additional information
    
    Returns:
    class_acc_list: list of classification accuracies for each fold
    negone_acc_list: list of classification accuracies for class -1 for each fold
    one_acc_list: list of classification accuracies for class 1 for each fold
    negone_barys: list of FGW barycenters for class -1 for each fold
    one_barys: list of FGW barycenters for class 1 for each fold
    f1_score_list: list of F1 scores for each fold"""

    class_acc_list = [
    ]  # list to store classification accuracies from each fold
    negone_acc_list = []  # list to store class -1 accuracies from each fold
    one_acc_list = []  # list to store class 1 accuracies from each fold
    f1_score_list = []  # list to store F1 scores from each fold
    negone_barys = []  # list to store class -1 barycenters from each fold
    one_barys = []  # list to store class 1 barycenters from each fold

    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(k_folds.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log is True:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
        X_train, X_test = [distance_matrices[idx] for idx in train_index
                           ], [distance_matrices[idx] for idx in test_index]
        y_train, y_test = [labels[idx] for idx in train_index
                           ], [labels[idx] for idx in test_index]
        X_train_negone = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        X_train_one = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == 1
        ]

        # Compute the GW barycenter for each category
        # Get node attributes for class -1
        features_negone = []
        negone_indices = [
            train_index[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        for graph_idx in negone_indices:
            graph = graphs_list[graph_idx]
            graph_features = np.array(
                [graph.nodes[n]['attributes'] for n in graph.nodes()])
            features_negone.append(graph_features)

        # Create uniform distributions for each graph
        ps_negone = [
            np.ones(len(features)) / len(features)
            for features in features_negone
        ]
        lambdas_negone = np.ones(len(features_negone)) / len(features_negone)

        barycenter_features_negone, barycenter_matrix_negone, log_negone = ot.gromov.fgw_barycenters(
            N=num_nodes_negone,
            Ys=features_negone,
            Cs=X_train_negone,
            ps=ps_negone,
            lambdas=lambdas_negone,
            loss_fun='square_loss',
            random_state=seed,
            log=True)
        if log is True:
            print("GW barycenter for Class -1 graphs:",
                  barycenter_matrix_negone, f"Fold {i}")
        negone_barys.append(barycenter_matrix_negone)
        print("GW barycenter for Class -1 graphs computed")

        # Get node attributes for class 1
        features_one = []
        one_indices = [
            train_index[j] for j in range(len(X_train)) if y_train[j] == 1
        ]
        for graph_idx in one_indices:
            graph = graphs_list[graph_idx]
            graph_features = np.array(
                [graph.nodes[n]['attributes'] for n in graph.nodes()])
            features_one.append(graph_features)

        # Create uniform distributions for each graph
        ps_one = [
            np.ones(len(features)) / len(features)
            for features in features_one
        ]
        lambdas_one = np.ones(len(features_one)) / len(features_one)

        barycenter_features_one, barycenter_matrix_one, log_one = ot.gromov.fgw_barycenters(
            N=num_nodes_one,
            Ys=features_one,
            Cs=X_train_one,
            ps=ps_one,
            lambdas=lambdas_one,
            loss_fun='square_loss',
            random_state=seed,
            log=True)
        one_barys.append(barycenter_matrix_one)
        if log is True:
            print("GW barycenter for Class 1 graphs:",
                  barycenter_matrix_one, f"Fold {i}")
        print("GW barycenter for Class 1 graphs computed")

        # Compute metric cost matrix between barycenters
        M = ot.dist(barycenter_features_negone, barycenter_features_one)

        fgw, log_dists = ot.gromov.fused_gromov_wasserstein(
            M,
            barycenter_matrix_negone,
            barycenter_matrix_one,
            loss_fun='square_loss',
            log=True)
        print("Distance between barycenters: ", log_dists['fgw_dist'],
              f" Fold {i}")

        correct_count = 0
        correct_count_negone = 0
        correct_count_one = 0
        total_count_negone = 0
        total_count_one = 0

        predicted_labels = []
        true_labels = []
        total_samples = len(X_test)
        for idx, graph in enumerate(X_test):
            # Get node attributes for test graph
            test_features = [
                graphs_list[test_index[idx]].nodes[n]['attributes']
                for n in graphs_list[test_index[idx]].nodes()
            ]

            # Compute metric cost matrices
            M1 = ot.dist(np.array(test_features), barycenter_features_negone)
            M2 = ot.dist(np.array(test_features), barycenter_features_one)

            fgw_negone, log_negone = ot.gromov.fused_gromov_wasserstein(
                M1,
                graph,
                barycenter_matrix_negone,
                loss_fun='square_loss',
                log=True)
            fgw_one, log_one = ot.gromov.fused_gromov_wasserstein(
                M2,
                graph,
                barycenter_matrix_one,
                loss_fun='square_loss',
                log=True)
            ground_truth_label = y_test[idx]
            true_labels.append(ground_truth_label)
            # Check if classification is correct
            predicted_label = -1 if log_negone['fgw_dist'] < log_one[
                'fgw_dist'] else 1
            predicted_labels.append(predicted_label)
            if ground_truth_label == -1:
                total_count_negone += 1
                if predicted_label == ground_truth_label:
                    correct_count_negone += 1
            if ground_truth_label == 1:
                total_count_one += 1
                if predicted_label == ground_truth_label:
                    correct_count_one += 1
            if predicted_label == ground_truth_label:
                correct_count += 1
        # Calculate classification accuracy
        classification_accuracy = (correct_count / total_samples) * 100
        negone_accuracy = (correct_count_negone / total_count_negone) * 100
        one_accuracy = (correct_count_one / total_count_one) * 100

        # Calculate F1 scores
        average_type = "macro" if method == "kfold" else "weighted"
        f1_weighted = f1_score(
            true_labels, predicted_labels, average=average_type) * 100

        print(f"F1 Weighted: {f1_weighted}%, Fold {i}")
        print(f"Classification Accuracy: {classification_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class -1: {negone_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class 1: {one_accuracy}%",
              f"Fold {i}")
        class_acc_list.append(classification_accuracy)
        negone_acc_list.append(negone_accuracy)
        one_acc_list.append(one_accuracy)
        f1_score_list.append(f1_weighted)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)

    # Compute and print averages and standard deviations
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    avg_classification_acc = np.mean(class_acc_list)
    std_classification_acc = np.std(class_acc_list)
    print(
        f"Overall Classification Accuracy: {avg_classification_acc:.2f}% ± {std_classification_acc:.2f}%"
    )

    avg_negone_acc = np.mean(negone_acc_list)
    std_negone_acc = np.std(negone_acc_list)
    print(
        f"Class -1 Classification Accuracy: {avg_negone_acc:.2f}% ± {std_negone_acc:.2f}%"
    )

    avg_one_acc = np.mean(one_acc_list)
    std_one_acc = np.std(one_acc_list)
    print(
        f"Class 1 Classification Accuracy: {avg_one_acc:.2f}% ± {std_one_acc:.2f}%"
    )

    avg_f1_score = np.mean(f1_score_list)
    std_f1_score = np.std(f1_score_list)
    average_type = "macro" if method == "kfold" else "weighted"
    print(
        f"F1 Score ({average_type} avg): {avg_f1_score:.2f}% ± {std_f1_score:.2f}%"
    )

    return class_acc_list, negone_acc_list, one_acc_list, f1_score_list, negone_barys, one_barys


def fgw_barycenter_alg_alpha(distance_matrices,
                             graphs_list,
                             labels,
                             num_folds,
                             num_nodes_negone,
                             num_nodes_one,
                             alpha,
                             method,
                             seed=42,
                             log=False):
    """Implements FGW-NCC binary classification algorithm with variable number of nodes for the FGW barycenter of each class. Also includes alpha parameter for hyperparameter tuning.
    Parameters:
    distance_matrices: list of pairwise distance matrices
    graphs_list: list of graphs
    labels: list of labels for each graph
    num_folds: number of folds for cross-validation
    num_nodes_negone: number of nodes for the FGW barycenter of class -1
    num_nodes_one: number of nodes for the FGW barycenter of the class 1
    alpha: alpha parameter for FGW
    method: cross-validation method, either "kfold" (for k-fold
    CV) or "stratk" (for stratified k-fold CV)
    seed: random seed for reproducibility
    log: boolean flag to print additional information
    
    Returns:
    class_acc_list: list of classification accuracies for each fold
    negone_acc_list: list of classification accuracies for class -1 for each fold
    one_acc_list: list of classification accuracies for class 1 for each fold
    f1_score_list: list of F1 scores for each fold
    negone_barys: list of FGW barycenters for class -1 for each fold
    one_barys: list of FGW barycenters for class 1 for each fold
    """
    
    class_acc_list = [
    ]  # list to store classification accuracies from each fold
    f1_score_list = []  # list to store F1 scores from each fold
    negone_acc_list = []  # list to store class -1 accuracies from each fold
    one_acc_list = []  # list to store class 1 accuracies from each fold
    negone_barys = []  # list to store class -1 barycenters from each fold
    one_barys = []  # list to store class 1 barycenters from each fold

    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(k_folds.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log is True:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
        X_train, X_test = [distance_matrices[idx] for idx in train_index
                           ], [distance_matrices[idx] for idx in test_index]
        y_train, y_test = [labels[idx] for idx in train_index
                           ], [labels[idx] for idx in test_index]
        X_train_negone = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        X_train_one = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == 1
        ]

        # Compute the GW barycenter for each category
        # Get node attributes for class -1
        features_negone = []
        negone_indices = [
            train_index[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        for graph_idx in negone_indices:
            graph = graphs_list[graph_idx]
            graph_features = np.array(
                [graph.nodes[n]['attributes'] for n in graph.nodes()])
            features_negone.append(graph_features)

        # Create uniform distributions for each graph
        ps_negone = [
            np.ones(len(features)) / len(features)
            for features in features_negone
        ]
        lambdas_negone = np.ones(len(features_negone)) / len(features_negone)

        barycenter_features_negone, barycenter_matrix_negone, log_negone = ot.gromov.fgw_barycenters(
            N=num_nodes_negone,
            Ys=features_negone,
            Cs=X_train_negone,
            ps=ps_negone,
            lambdas=lambdas_negone,
            alpha=alpha,
            loss_fun='square_loss',
            random_state=seed,
            log=True)
        if log is True:
            print("GW barycenter for Class -1 graphs:",
                  barycenter_matrix_negone, f"Fold {i}")
        negone_barys.append(barycenter_matrix_negone)
        print("GW barycenter for Class -1 graphs computed")

        # Get node attributes for class 1
        features_one = []
        one_indices = [
            train_index[j] for j in range(len(X_train)) if y_train[j] == 1
        ]
        for graph_idx in one_indices:
            graph = graphs_list[graph_idx]
            graph_features = np.array(
                [graph.nodes[n]['attributes'] for n in graph.nodes()])
            features_one.append(graph_features)

        # Create uniform distributions for each graph
        ps_one = [
            np.ones(len(features)) / len(features)
            for features in features_one
        ]
        lambdas_one = np.ones(len(features_one)) / len(features_one)

        barycenter_features_one, barycenter_matrix_one, log_one = ot.gromov.fgw_barycenters(
            N=num_nodes_one,
            Ys=features_one,
            Cs=X_train_one,
            ps=ps_one,
            lambdas=lambdas_one,
            alpha=alpha,
            loss_fun='square_loss',
            random_state=seed,
            log=True)
        one_barys.append(barycenter_matrix_one)
        if log is True:
            print("GW barycenter for Class 1 graphs:",
                  barycenter_matrix_one, f"Fold {i}")
        print("GW barycenter for Class 1 graphs computed")

        # Compute metric cost matrix between barycenters
        M = ot.dist(barycenter_features_negone, barycenter_features_one)

        fgw, log_dists = ot.gromov.fused_gromov_wasserstein(
            M,
            barycenter_matrix_negone,
            barycenter_matrix_one,
            loss_fun='square_loss',
            alpha=alpha,
            log=True)
        print("Distance between barycenters: ", log_dists['fgw_dist'],
              f" Fold {i}")

        correct_count = 0
        correct_count_negone = 0
        correct_count_one = 0
        total_count_negone = 0
        total_count_one = 0

        predicted_labels = []
        true_labels = []
        total_samples = len(X_test)
        for idx, graph in enumerate(X_test):
            # Get node attributes for test graph
            test_features = [
                graphs_list[test_index[idx]].nodes[n]['attributes']
                for n in graphs_list[test_index[idx]].nodes()
            ]

            # Compute metric cost matrices
            M1 = ot.dist(np.array(test_features), barycenter_features_negone)
            M2 = ot.dist(np.array(test_features), barycenter_features_one)

            fgw_negone, log_negone = ot.gromov.fused_gromov_wasserstein(
                M1,
                graph,
                barycenter_matrix_negone,
                loss_fun='square_loss',
                alpha=alpha,
                log=True)
            fgw_one, log_one = ot.gromov.fused_gromov_wasserstein(
                M2,
                graph,
                barycenter_matrix_one,
                loss_fun='square_loss',
                alpha=alpha,
                log=True)
            ground_truth_label = y_test[idx]
            true_labels.append(ground_truth_label)
            # Check if classification is correct
            predicted_label = -1 if log_negone['fgw_dist'] < log_one[
                'fgw_dist'] else 1
            predicted_labels.append(predicted_label)
            if ground_truth_label == -1:
                total_count_negone += 1
                if predicted_label == ground_truth_label:
                    correct_count_negone += 1
            if ground_truth_label == 1:
                total_count_one += 1
                if predicted_label == ground_truth_label:
                    correct_count_one += 1
            if predicted_label == ground_truth_label:
                correct_count += 1
        # Calculate classification accuracy
        classification_accuracy = (correct_count / total_samples) * 100
        negone_accuracy = (correct_count_negone / total_count_negone) * 100
        one_accuracy = (correct_count_one / total_count_one) * 100
        negone_acc_list.append(negone_accuracy)
        one_acc_list.append(one_accuracy)

        # Calculate F1 scores
        average_type = "macro" if method == "kfold" else "weighted"
        f1_weighted = f1_score(
            true_labels, predicted_labels, average=average_type) * 100
        f1_score_list.append(f1_weighted)

        print(f"F1 Score ({average_type} avg): {f1_weighted}%, Fold {i}")
        print(f"Classification Accuracy: {classification_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class -1: {negone_accuracy}%",
              f"Fold {i}")
        print(f"Classification Accuracy for Class 1: {one_accuracy}%",
              f"Fold {i}")
        class_acc_list.append(classification_accuracy)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)

    # Compute and print averages and standard deviations
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    avg_classification_acc = np.mean(class_acc_list)
    std_classification_acc = np.std(class_acc_list)
    print(
        f"Overall Classification Accuracy: {avg_classification_acc:.2f}% ± {std_classification_acc:.2f}%"
    )

    avg_negone_acc = np.mean(negone_acc_list)
    std_negone_acc = np.std(negone_acc_list)
    print(
        f"Class -1 Classification Accuracy: {avg_negone_acc:.2f}% ± {std_negone_acc:.2f}%"
    )

    avg_one_acc = np.mean(one_acc_list)
    std_one_acc = np.std(one_acc_list)
    print(
        f"Class 1 Classification Accuracy: {avg_one_acc:.2f}% ± {std_one_acc:.2f}%"
    )

    avg_f1_score = np.mean(f1_score_list)
    std_f1_score = np.std(f1_score_list)
    average_type = "macro" if method == "kfold" else "weighted"
    print(
        f"F1 Score ({average_type} avg): {avg_f1_score:.2f}% ± {std_f1_score:.2f}%"
    )

    return class_acc_list, negone_acc_list, one_acc_list, f1_score_list, negone_barys, one_barys


"""
GW-kNN, gLGW-kNN, and fGW-kNN algorithms
Citation for gLGW codes: https://github.com/Gorgotha/LGW
lgw_procedure, LGW_graph, and LGW_eucl are taken directly from this code.
k_folds_glgw_knn is a modified function from the above code. Everything else is my work.
"""

def gw_knn(distance_matrices,
           labels,
           num_folds,
           k_list,
           method,
           seed=42,
           log_knn=False):
    """Function to perform k-nearest neighbors classification using Gromov-Wasserstein distances.
    Parameters:
    - distance_matrices: List of distance matrices.
    - labels: List of labels corresponding to the distance matrices.
    - num_folds: Number of folds for cross-validation.
    - k_list: List of k values to test.
    - method: Cross-validation method, either 'kfold' or 'stratk'.
    - seed: Random seed for reproducibility. Defaults to 42.
    - log_knn: Boolean flag to log k-NN details. Defaults to False.

    Returns:
    - class_acc_dict: Dictionary storing accuracies for each k.
    - by_class_acc_dict: Dictionary storing accuracies by class for each k.
    - f1_score_dict: Dictionary storing F1 scores for each k.
    - best_k_info: Dictionary storing information about the best k value.
    """
    class_acc_dict = {
        k: []
        for k in k_list
    }  # dictionary to store accuracies for each k
    by_class_acc_dict = {
        k: {
            1: [],
            -1: []
        }
        for k in k_list
    }  # store accuracies by class for each k
    f1_score_dict = {
        k: []
        for k in k_list
    }  # dictionary to store F1 scores for each k
    assert method in ["kfold", "stratk"]
    if method == "kfold":
        skf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        skf = StratifiedKFold(n_splits=num_folds,
                              shuffle=True,
                              random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()

    for i, (train_index,
            test_index) in enumerate(skf.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log_knn:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

        X_train = [distance_matrices[idx] for idx in train_index]
        y_train = [labels[idx] for idx in train_index]
        X_test = [distance_matrices[idx] for idx in test_index]
        y_test = [labels[idx] for idx in test_index]

        # Compute all pairwise distances once per fold
        all_distances = []
        for test_graph, ground_truth_label in zip(X_test, y_test):
            distances = []
            for train_graph, train_label in zip(X_train, y_train):
                gw, log = ot.gromov.gromov_wasserstein(test_graph,
                                                       train_graph,
                                                       loss_fun='square_loss',
                                                       log=True,
                                                       random_seed=seed)
                distances.append((log['gw_dist'], train_label))
            distances.sort(key=lambda x: x[0])  # Sort based on distances
            all_distances.append((distances, ground_truth_label))

        # For each k value, perform classification
        for k in k_list:
            correct_count = 0
            class_correct = {1: 0, -1: 0}
            class_total = {1: 0, -1: 0}
            y_true = []
            y_pred = []

            for distances, ground_truth_label in all_distances:
                # Take the k nearest neighbors
                k_nearest_neighbors = distances[:k]
                # Determine the most common label
                labels_counts = {}
                for _, label in k_nearest_neighbors:
                    labels_counts[label] = labels_counts.get(label, 0) + 1
                predicted_label = max(labels_counts, key=labels_counts.get)

                y_true.append(ground_truth_label)
                y_pred.append(predicted_label)

                if predicted_label == ground_truth_label:
                    correct_count += 1
                    class_correct[ground_truth_label] += 1
                class_total[ground_truth_label] += 1

            # Calculate overall and by-class accuracies
            classification_accuracy = (correct_count / len(X_test)) * 100
            class_acc_dict[k].append(classification_accuracy)

            # Calculate F1 score with appropriate averaging
            if method == "kfold":
                f1 = f1_score(y_true, y_pred, average='macro') * 100
            else:  # stratk
                f1 = f1_score(y_true, y_pred, average='weighted') * 100
            f1_score_dict[k].append(f1)

            # Calculate and store by-class accuracies
            for class_label in [1, -1]:
                if class_total[class_label] > 0:
                    class_accuracy = (class_correct[class_label] /
                                      class_total[class_label]) * 100
                    by_class_acc_dict[k][class_label].append(class_accuracy)
                    print(
                        f"k={k}, Class {class_label} Accuracy: {class_accuracy}%, Fold {i}"
                    )

            print(
                f"k={k}, Overall Classification Accuracy: {classification_accuracy}%, Fold {i}"
            )
            average_type = "macro" if method == "kfold" else "weighted"
            print(f"k={k}, F1 Score ({average_type} avg): {f1:.2f}%, Fold {i}")

    end = time.time()
    print("Total runtime, seconds:", end - start)

    # Print accuracy stats for each k and print best overall k
    best_k = k_list[0]
    best_acc = 0
    best_f1 = 0
    k_avg_accuracies = {}
    k_avg_f1_scores = {}
    k_std_accuracies = {}
    k_std_f1_scores = {}

    average_type = "macro" if method == "kfold" else "weighted"

    for k in k_list:
        avg_acc = sum(class_acc_dict[k]) / len(class_acc_dict[k])
        avg_f1 = sum(f1_score_dict[k]) / len(f1_score_dict[k])

        # Calculate standard deviations
        std_acc = np.std(class_acc_dict[k])
        std_f1 = np.std(f1_score_dict[k])
        std_class_1 = np.std(by_class_acc_dict[k][1])
        std_class_neg1 = np.std(by_class_acc_dict[k][-1])

        k_avg_accuracies[k] = {
            'overall':
            avg_acc,
            'class_1':
            sum(by_class_acc_dict[k][1]) / len(by_class_acc_dict[k][1]),
            'class_-1':
            sum(by_class_acc_dict[k][-1]) / len(by_class_acc_dict[k][-1])
        }
        k_avg_f1_scores[k] = avg_f1
        k_std_accuracies[k] = {
            'overall': std_acc,
            'class_1': std_class_1,
            'class_-1': std_class_neg1
        }
        k_std_f1_scores[k] = std_f1

        print(f"\nk={k} Average Overall Accuracy: {avg_acc:.2f}%")
        print(f"k={k} Average F1 Score ({average_type} avg): {avg_f1:.2f}%")
        for class_label in [1, -1]:
            print(
                f"k={k} Average Class {class_label} Accuracy: {k_avg_accuracies[k][f'class_{class_label}']:.2f}%"
            )

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_k = k

        if avg_f1 > best_f1:
            best_f1 = avg_f1

    print(f"\nBest performance achieved with k={best_k}:")
    print(
        f"Overall Accuracy: {k_avg_accuracies[best_k]['overall']:.2f}% ± {k_std_accuracies[best_k]['overall']:.2f}%"
    )
    print(
        f"F1 Score ({average_type} avg): {k_avg_f1_scores[best_k]:.2f}% ± {k_std_f1_scores[best_k]:.2f}%"
    )
    print(
        f"Class 1 Accuracy: {k_avg_accuracies[best_k]['class_1']:.2f}% ± {k_std_accuracies[best_k]['class_1']:.2f}%"
    )
    print(
        f"Class -1 Accuracy: {k_avg_accuracies[best_k]['class_-1']:.2f}% ± {k_std_accuracies[best_k]['class_-1']:.2f}%"
    )

    # Prepare best k information to return
    best_k_info = {
        'best_k': best_k,
        'best_overall_accuracy': k_avg_accuracies[best_k]['overall'],
        'best_f1_score': k_avg_f1_scores[best_k],
        'best_class_1_accuracy': k_avg_accuracies[best_k]['class_1'],
        'best_class_minus1_accuracy': k_avg_accuracies[best_k]['class_-1']
    }

    return class_acc_dict, by_class_acc_dict, f1_score_dict, best_k_info


def lgw_procedure(M_ref,
                  height_ref,
                  posns,
                  Ms,
                  heights,
                  max_iter=1000,
                  mode="euclidean"):
    """
  Computes the generalized linear Gromov-Wasserstein distance between similarity matrices.
  Parameters:
  - M_ref: reference barycenter
  - height_ref: distribution/weights in the source space
  - posns: If mode="euclidean", posns is a list of positions to compute the Euclidean embeddings. If mode="graph", enter None.
  - Ms: list of similarity matrices (a matrix of measures)
  - heights: distribution/weights in the target space
  - max_iter: maximum number of iterations (default: 1000)
  - mode: "euclidean" or "graph" (default: "euclidean"). Determines which norm to use for the distance computation. Choice of norm depends on type of data.
  Output:
  - lgw: a matrix containing the pairwise generalized linear Gromov-Wasserstein distances
  - et-st: total time
  """
    assert mode in ["euclidean", "graph"]
    N = len(Ms)

    Ps = []  #GW Plans
    Ts = []  #barycentric projections
    st = time.time()
    for i in range(0, N):
        #GW computation
        P = ot.gromov.gromov_wasserstein(M_ref,
                                         Ms[i],
                                         height_ref,
                                         heights[i],
                                         "square_loss",
                                         log=True)[0]
        Ps.append(P)

        #euclidean barycentric projection
        if mode == "euclidean":
            T = (np.divide(P.T, height_ref).T).dot(posns[i])
        #generalized barycentric projection
        else:
            T = []
            k = len(Ms[i])
            k_ref = len(M_ref)
            for v in range(k_ref):
                barycentricity = []
                weights = P[v] / height_ref[v]
                for w in range(k):
                    breakBool = False
                    if weights[w] == 1:
                        bary = w
                        breakBool = True
                        break
                    barycentricity.append(weights.dot(Ms[i][w]**2))
                if breakBool:
                    T.append(bary)
                else:
                    bary = np.argmin(barycentricity)
                    T.append(bary)
        Ts.append(T)

    #LGW computation
    lgw = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if mode == "euclidean":
                lgw[i, j] = LGW_eucl(Ts[i], Ts[j], height_ref)
            else:
                lgw[i, j] = LGW_graph(Ts[i], Ts[j], Ms[i], Ms[j], height_ref)
    lgw += lgw.T
    et = time.time()
    return lgw, et - st


"""
The paper: https://arxiv.org/pdf/2112.11964
The relevant sections on the barycenter computations are sections III C and IV C. The difference between the two functions ocurrs due to differences in how the barycentric projections are computed, depending on the type of data you have.
"""


def LGW_graph(T1, T2, D1, D2, sigma):
    """
  Works directly with the graph distance matrices D1, D2 and uses the transport maps T1, T2 to select relevant entries from these matrices. It then computes the distance between graphs using their matrix representations.
  """
    return np.sqrt(
        np.sum(
            np.multiply((D1[T1].T[T1].T - D2[T2].T[T2].T)**2,
                        np.outer(sigma, sigma))))


def LGW_eucl(T1, T2, sigma, normalized=False):
    """
  First converts the transport maps T1, T2 into Euclidean distance matrices using ot.dist, with an optional normalization. Then it compares these Euclidean distances. In lgw_procedure, the "euclidean" mode will additionally require position data to compute Euclidean embeddings.
  """
    M1 = ot.dist(T1, T1, metric="euclidean")
    M2 = ot.dist(T2, T2, metric="euclidean")
    if normalized:
        M1 = M1 / np.max(M1)
        M2 = M2 / np.max(M2)
    return np.sqrt(np.sum(np.multiply((M1 - M2)**2, np.outer(sigma, sigma))))


def glgw_knn(lgw_matrix,
             labels,
             train_index,
             test_index,
             k_list,
             method,
             log_knn=False):
    """
        Computes the k-nearest neighbors for a given graph using the LGW distance.
        Parameters:
        - lgw_matrix: the LGW distance matrix containing all the pairwise LGW distances between graphs
        - labels: the labels of the graphs
        - train_index: the indices of the graphs from the training set
        - test_index: the indices of the graphs from the test set
        - k_list: list of k values to evaluate
        - method: Cross-validation method, either 'kfold' or 'stratk'
        - log_knn: whether to print the classification accuracy log (default: False)
        Output:
        - classification accuracy: the classification accuracy of the kNN algorithm
    """
    class_acc_dict = {k: [] for k in k_list}
    by_class_acc_dict = {k: {1: [], -1: []} for k in k_list}
    f1_score_dict = {k: [] for k in k_list}

    X_train, y_train = lgw_matrix[train_index], [
        labels[idx] for idx in train_index
    ]
    X_test, y_test = lgw_matrix[test_index], [
        labels[idx] for idx in test_index
    ]

    # Initialize counters for each k
    correct_counts = {k: 0 for k in k_list}
    class_correct = {k: {1: 0, -1: 0} for k in k_list}
    class_total = {k: {1: 0, -1: 0} for k in k_list}
    predictions = {k: [] for k in k_list}

    for test_idx, ground_truth_label in zip(test_index, y_test):
        distances = [(lgw_matrix[test_idx, idx], y_train[j])
                     for j, idx in enumerate(train_index)]
        distances.sort(key=lambda x: x[0])

        for k in k_list:
            k_nearest_neighbors = distances[:k]
            labels_counts = {}
            for _, label in k_nearest_neighbors:
                labels_counts[label] = labels_counts.get(label, 0) + 1
            predicted_label = max(labels_counts, key=labels_counts.get)

            predictions[k].append(predicted_label)

            if predicted_label == ground_truth_label:
                correct_counts[k] += 1
                class_correct[k][ground_truth_label] += 1
            class_total[k][ground_truth_label] += 1

    # Calculate accuracies and F1 scores after all predictions
    average_type = "macro" if method == "kfold" else "weighted"

    for k in k_list:
        # Overall accuracy
        classification_accuracy = (correct_counts[k] / len(test_index)) * 100
        class_acc_dict[k].append(classification_accuracy)

        # Calculate F1 score with appropriate averaging
        if method == "kfold":
            f1 = f1_score(y_test, predictions[k], average='macro') * 100
        else:  # stratk
            f1 = f1_score(y_test, predictions[k], average='weighted') * 100
        f1_score_dict[k].append(f1)

        # By-class accuracies
        for class_label in [1, -1]:
            if class_total[k][class_label] > 0:
                class_accuracy = (class_correct[k][class_label] /
                                  class_total[k][class_label]) * 100
                by_class_acc_dict[k][class_label].append(class_accuracy)
                if log_knn:
                    print(
                        f"k={k}, Class {class_label} Accuracy: {class_accuracy}%"
                    )

        if log_knn:
            print(
                f"k={k}, Overall Classification Accuracy: {classification_accuracy}%"
            )
            print(f"k={k}, F1 Score ({average_type} avg): {f1:.2f}%")

    return class_acc_dict, by_class_acc_dict, f1_score_dict


def k_folds_glgw_knn(X, y, k_bary, Ms, heights, num_folds, seed, k_knn_list,
                     method):
    """
    Implements k-fold cross-validation in conjunction with the gLGW-kNN algorithm.
    Parameters:
    - X: the input data
    - y: the labels of the data
    - k_bary: the number of points to use for the reference barycenter in the LGW computation
    - Ms: the list of similarity matrices (a matrix of measures)
    - heights: the distribution/weights in the target space
    - num_folds: the number of folds to use for cross-validation
    - seed: the seed for the random number generator
    - k_knn_list: the list of k values to use for k-fold cross-validation
    - method: choice of k-fold cross validation (kfold) or stratified k-fold cross validation (stratk)

    Returns:
    - class_acc_dict: dictionary storing accuracies for each k
    - by_class_acc_dict: dictionary storing accuracies by class for each k
    - f1_score_dict: dictionary storing F1 scores for each k
    - best_k_info: dictionary storing information about the best k value
    """
    np.random.seed(seed)
    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")

    # Initialize storage for all metrics
    class_acc_dict = {k: [] for k in k_knn_list}
    by_class_acc_dict = {k: {1: [], -1: []} for k in k_knn_list}
    f1_score_dict = {k: [] for k in k_knn_list}

    classes = [-1, 1]
    fold_num = 0

    for train_index, test_index in k_folds.split(X, y):
        print(f"Fold {fold_num}:")
        X_train, X_test = [Ms[idx] for idx in train_index
                           ], [Ms[idx] for idx in test_index]
        y_train, y_test = np.array([y[idx] for idx in train_index]), np.array(
            [y[idx] for idx in test_index])
        # Compute reference barycenter for the training set
        idx_bary = []
        for i in classes:
            # Find the indices of instances with label i
            indices = np.where(y_train == i)[0]
            if indices.size > 0:
                idx_bary.extend(
                    np.random.choice(train_index[indices],
                                     size=min(5, len(indices)),
                                     replace=False))
        bary_start_time = time.time()
        # Initialize reference matrix with k_bary size
        M_ref = np.zeros((k_bary, k_bary))
        selected_Ms = [Ms[i] for i in idx_bary]
        selected_heights = [heights[i] for i in idx_bary]
        lambdas = ot.unif(len(idx_bary))
        M_ref = ot.gromov.gromov_barycenters(k_bary,
                                             Cs=selected_Ms,
                                             ps=selected_heights,
                                             p=ot.unif(k_bary),
                                             lambdas=lambdas,
                                             loss_fun='square_loss',
                                             max_iter=200,
                                             tol=1e-12,
                                             random_state=seed)
        bary_computation_time = time.time() - bary_start_time
        print(
            f"Time taken for reference barycenter computation: {bary_computation_time:.4f} seconds"
        )
        height_ref = ot.unif(k_bary)
        lgw_matrix, lgw_time = lgw_procedure(M_ref,
                                             height_ref,
                                             None,
                                             Ms,
                                             heights,
                                             mode="graph")

        print(f"Time taken for LGW computation: {lgw_time:.4f} seconds")

        # Evaluate for each k in k_knn_list
        fold_class_acc, fold_by_class_acc, fold_f1_scores = glgw_knn(
            lgw_matrix,
            y,
            train_index,
            test_index,
            k_knn_list,
            method,
            log_knn=True)

        # Accumulate results
        for k in k_knn_list:
            class_acc_dict[k].extend(fold_class_acc[k])
            f1_score_dict[k].extend(fold_f1_scores[k])
            for class_label in [1, -1]:
                by_class_acc_dict[k][class_label].extend(
                    fold_by_class_acc[k][class_label])

        fold_num += 1

    # Find best k and print stats
    best_k = k_knn_list[0]
    best_acc = 0
    best_f1 = 0
    k_avg_accuracies = {}
    k_avg_f1_scores = {}
    k_std_accuracies = {}
    k_std_f1_scores = {}

    average_type = "macro" if method == "kfold" else "weighted"

    for k in k_knn_list:
        avg_acc = sum(class_acc_dict[k]) / len(class_acc_dict[k])
        avg_f1 = sum(f1_score_dict[k]) / len(f1_score_dict[k])

        # Calculate standard deviations
        std_acc = np.std(class_acc_dict[k])
        std_f1 = np.std(f1_score_dict[k])
        std_class_1 = np.std(by_class_acc_dict[k][1])
        std_class_neg1 = np.std(by_class_acc_dict[k][-1])

        k_avg_accuracies[k] = {
            'overall':
            avg_acc,
            'class_1':
            sum(by_class_acc_dict[k][1]) / len(by_class_acc_dict[k][1]),
            'class_-1':
            sum(by_class_acc_dict[k][-1]) / len(by_class_acc_dict[k][-1])
        }
        k_avg_f1_scores[k] = avg_f1

        k_std_accuracies[k] = {
            'overall': std_acc,
            'class_1': std_class_1,
            'class_-1': std_class_neg1
        }
        k_std_f1_scores[k] = std_f1

        print(f"\nk={k} Average Overall Accuracy: {avg_acc:.2f}%")
        print(f"k={k} Average F1 Score ({average_type} avg): {avg_f1:.2f}%")
        for class_label in [1, -1]:
            print(
                f"k={k} Average Class {class_label} Accuracy: {k_avg_accuracies[k][f'class_{class_label}']:.2f}%"
            )

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_k = k

        if avg_f1 > best_f1:
            best_f1 = avg_f1

    print(f"\nBest performance achieved with k={best_k}:")
    print(
        f"Overall Accuracy: {k_avg_accuracies[best_k]['overall']:.2f}% ± {k_std_accuracies[best_k]['overall']:.2f}%"
    )
    print(
        f"F1 Score ({average_type} avg): {k_avg_f1_scores[best_k]:.2f}% ± {k_std_f1_scores[best_k]:.2f}%"
    )
    print(
        f"Class 1 Accuracy: {k_avg_accuracies[best_k]['class_1']:.2f}% ± {k_std_accuracies[best_k]['class_1']:.2f}%"
    )
    print(
        f"Class -1 Accuracy: {k_avg_accuracies[best_k]['class_-1']:.2f}% ± {k_std_accuracies[best_k]['class_-1']:.2f}%"
    )

    # Prepare best k information to return
    best_k_info = {
        'best_k': best_k,
        'best_overall_accuracy': k_avg_accuracies[best_k]['overall'],
        'best_f1_score': k_avg_f1_scores[best_k],
        'best_class_1_accuracy': k_avg_accuracies[best_k]['class_1'],
        'best_class_minus1_accuracy': k_avg_accuracies[best_k]['class_-1']
    }

    return class_acc_dict, by_class_acc_dict, f1_score_dict, best_k_info


def fgw_knn(distance_matrices,
            graphs_list,
            labels,
            num_folds,
            k_list,
            method,
            seed=42,
            log_knn=False):
    """Function to perform k-nearest neighbors classification using Fused Gromov-Wasserstein distances.
    Parameters:
    - distance_matrices: list of pairwise distance matrices
    - graphs_list: list of graphs
    - labels: list of labels for each graph
    - num_folds: number of folds for cross-validation
    - method: cross-validation method, either "kfold" (for k-fold
    CV) or "stratk" (for stratified k-fold CV)
    - seed: random seed for reproducibility
    - log: boolean flag to print additional information

    Returns:
    - class_acc_dict: dictionary storing accuracies for each k
    - by_class_acc_dict: dictionary storing accuracies by class for each k
    - f1_score_dict: dictionary storing F1 scores for each k
    - best_k_info: dictionary storing information about the best k value
    """
    class_acc_dict = {
        k: []
        for k in k_list
    }  # dictionary to store accuracies for each k
    by_class_acc_dict = {
        k: {
            1: [],
            -1: []
        }
        for k in k_list
    }  # store accuracies by class for each k

    f1_score_dict = {
        k: []
        for k in k_list
    }  # dictionary to store F1 scores for each k

    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()

    for i, (train_index,
            test_index) in enumerate(k_folds.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log_knn:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

        X_train = [distance_matrices[idx] for idx in train_index]
        y_train = [labels[idx] for idx in train_index]
        X_test = [distance_matrices[idx] for idx in test_index]
        y_test = [labels[idx] for idx in test_index]

        all_distances = []
        for test_idx, (test_graph,
                       ground_truth_label) in enumerate(zip(X_test, y_test)):
            # Get test graph features
            test_graph_obj = graphs_list[test_index[test_idx]]
            test_features = np.array([
                test_graph_obj.nodes[n]['attributes']
                for n in test_graph_obj.nodes()
            ])

            distances = []
            for train_idx, (train_graph,
                            train_label) in enumerate(zip(X_train, y_train)):
                # Get train graph features
                train_graph_obj = graphs_list[train_index[train_idx]]
                train_features = np.array([
                    train_graph_obj.nodes[n]['attributes']
                    for n in train_graph_obj.nodes()
                ])

                # Compute metric cost matrix between test and train features
                M = ot.dist(test_features, train_features)

                # Compute fGW distance using the M for this pair
                fgw, log = ot.gromov.fused_gromov_wasserstein(
                    M,
                    test_graph,
                    train_graph,
                    loss_fun='square_loss',
                    log=True,
                    random_seed=seed)
                distances.append((log['fgw_dist'], train_label))
            distances.sort(key=lambda x: x[0])  # Sort based on distances
            all_distances.append((distances, ground_truth_label))

        # For each k value, perform classification
        for k in k_list:
            correct_count = 0
            class_correct = {1: 0, -1: 0}
            class_total = {1: 0, -1: 0}
            y_true = []
            y_pred = []

            for distances, ground_truth_label in all_distances:
                # Take the k nearest neighbors
                k_nearest_neighbors = distances[:k]
                # Determine the most common label
                labels_counts = {}
                for _, label in k_nearest_neighbors:
                    labels_counts[label] = labels_counts.get(label, 0) + 1
                predicted_label = max(labels_counts, key=labels_counts.get)

                y_true.append(ground_truth_label)
                y_pred.append(predicted_label)

                if predicted_label == ground_truth_label:
                    correct_count += 1
                    class_correct[ground_truth_label] += 1
                class_total[ground_truth_label] += 1

            # Calculate overall and by-class accuracies
            classification_accuracy = (correct_count / len(X_test)) * 100
            class_acc_dict[k].append(classification_accuracy)

            # Calculate F1 score with appropriate averaging
            if method == "kfold":
                f1 = f1_score(y_true, y_pred, average='macro') * 100
            else:  # stratk
                f1 = f1_score(y_true, y_pred, average='weighted') * 100
            f1_score_dict[k].append(f1)

            # Calculate and store by-class accuracies
            for class_label in [1, -1]:
                if class_total[class_label] > 0:
                    class_accuracy = (class_correct[class_label] /
                                      class_total[class_label]) * 100
                    by_class_acc_dict[k][class_label].append(class_accuracy)
                    print(
                        f"k={k}, Class {class_label} Accuracy: {class_accuracy}%, Fold {i}"
                    )

            print(
                f"k={k}, Overall Classification Accuracy: {classification_accuracy}%, Fold {i}"
            )
            average_type = "macro" if method == "kfold" else "weighted"
            print(f"k={k}, F1 Score ({average_type} avg): {f1:.2f}%, Fold {i}")

    end = time.time()
    print("Total runtime, seconds:", end - start)

    # Print accuracy stats for each k and print best overall k
    best_k = k_list[0]
    best_acc = 0
    best_f1 = 0
    k_avg_accuracies = {}
    k_avg_f1_scores = {}
    k_std_accuracies = {}
    k_std_f1_scores = {}

    average_type = "macro" if method == "kfold" else "weighted"

    for k in k_list:
        avg_acc = sum(class_acc_dict[k]) / len(class_acc_dict[k])
        avg_f1 = sum(f1_score_dict[k]) / len(f1_score_dict[k])

        # Calculate standard deviations
        std_acc = np.std(class_acc_dict[k])
        std_f1 = np.std(f1_score_dict[k])
        std_class_1 = np.std(by_class_acc_dict[k][1])
        std_class_neg1 = np.std(by_class_acc_dict[k][-1])

        k_avg_accuracies[k] = {
            'overall':
            avg_acc,
            'class_1':
            sum(by_class_acc_dict[k][1]) / len(by_class_acc_dict[k][1]),
            'class_-1':
            sum(by_class_acc_dict[k][-1]) / len(by_class_acc_dict[k][-1])
        }
        k_avg_f1_scores[k] = avg_f1

        k_std_accuracies[k] = {
            'overall': std_acc,
            'class_1': std_class_1,
            'class_-1': std_class_neg1
        }
        k_std_f1_scores[k] = std_f1

        print(f"\nk={k} Average Overall Accuracy: {avg_acc:.2f}%")
        print(f"k={k} Average F1 Score ({average_type} avg): {avg_f1:.2f}%")
        for class_label in [1, -1]:
            print(
                f"k={k} Average Class {class_label} Accuracy: {k_avg_accuracies[k][f'class_{class_label}']:.2f}%"
            )

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_k = k

        if avg_f1 > best_f1:
            best_f1 = avg_f1

    print(f"\nBest performance achieved with k={best_k}:")
    print(
        f"Overall Accuracy: {k_avg_accuracies[best_k]['overall']:.2f}% ± {k_std_accuracies[best_k]['overall']:.2f}%"
    )
    print(
        f"F1 Score ({average_type} avg): {k_avg_f1_scores[best_k]:.2f}% ± {k_std_f1_scores[best_k]:.2f}%"
    )
    print(
        f"Class 1 Accuracy: {k_avg_accuracies[best_k]['class_1']:.2f}% ± {k_std_accuracies[best_k]['class_1']:.2f}%"
    )
    print(
        f"Class -1 Accuracy: {k_avg_accuracies[best_k]['class_-1']:.2f}% ± {k_std_accuracies[best_k]['class_-1']:.2f}%"
    )

    # Prepare best k information to return
    best_k_info = {
        'best_k': best_k,
        'best_overall_accuracy': k_avg_accuracies[best_k]['overall'],
        'best_f1_score': k_avg_f1_scores[best_k],
        'best_class_1_accuracy': k_avg_accuracies[best_k]['class_1'],
        'best_class_minus1_accuracy': k_avg_accuracies[best_k]['class_-1']
    }

    return class_acc_dict, by_class_acc_dict, f1_score_dict, best_k_info


def fgw_knn_alpha(distance_matrices,
                  graphs_list,
                  labels,
                  num_folds,
                  k_list,
                  method,
                  alpha,
                  seed=42,
                  log_knn=False):
    """Function to perform k-nearest neighbors classification using Fused Gromov-Wasserstein distances. Also includes alpha parameter for hyperparameter tuning.
    Parameters:
    - distance_matrices: list of pairwise distance matrices
    - graphs_list: list of graphs
    - labels: list of labels for each graph
    - num_folds: number of folds for cross-validation
    - k_list: list of k values to test
    - method: cross-validation method, either "kfold" (for k-fold
    CV) or "stratk" (for stratified k-fold CV)
    - alpha: alpha parameter for fGW
    - seed: random seed for reproducibility
    - log: boolean flag to print additional information

    Returns:
    - class_acc_dict: dictionary storing accuracies for each k
    - by_class_acc_dict: dictionary storing accuracies by class for each k
    - f1_score_dict: dictionary storing F1 scores for each k
    - best_k_info: dictionary storing information about the best k value
    """
    class_acc_dict = {
        k: []
        for k in k_list
    }  # dictionary to store accuracies for each k
    by_class_acc_dict = {
        k: {
            1: [],
            -1: []
        }
        for k in k_list
    }  # store accuracies by class for each k

    f1_score_dict = {
        k: []
        for k in k_list
    }  # dictionary to store F1 scores for each k

    assert method in ["kfold", "stratk"]
    if method == "kfold":
        k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print("k-fold CV selected")
    else:
        k_folds = StratifiedKFold(n_splits=num_folds,
                                  shuffle=True,
                                  random_state=seed)
        print("Stratified k-fold CV selected")
    start = time.time()

    for i, (train_index,
            test_index) in enumerate(k_folds.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log_knn:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

        X_train = [distance_matrices[idx] for idx in train_index]
        y_train = [labels[idx] for idx in train_index]
        X_test = [distance_matrices[idx] for idx in test_index]
        y_test = [labels[idx] for idx in test_index]

        all_distances = []
        for test_idx, (test_graph,
                       ground_truth_label) in enumerate(zip(X_test, y_test)):
            # Get test graph features
            test_graph_obj = graphs_list[test_index[test_idx]]
            test_features = np.array([
                test_graph_obj.nodes[n]['attributes']
                for n in test_graph_obj.nodes()
            ])

            distances = []
            for train_idx, (train_graph,
                            train_label) in enumerate(zip(X_train, y_train)):
                # Get train graph features
                train_graph_obj = graphs_list[train_index[train_idx]]
                train_features = np.array([
                    train_graph_obj.nodes[n]['attributes']
                    for n in train_graph_obj.nodes()
                ])

                # Compute metric cost matrix between test and train features
                M = ot.dist(test_features, train_features)

                # Compute fGW distance using the M for this pair
                fgw, log = ot.gromov.fused_gromov_wasserstein(
                    M,
                    test_graph,
                    train_graph,
                    loss_fun='square_loss',
                    alpha=alpha,
                    log=True,
                    random_seed=seed)
                distances.append((log['fgw_dist'], train_label))
            distances.sort(key=lambda x: x[0])  # Sort based on distances
            all_distances.append((distances, ground_truth_label))

        # For each k value, perform classification
        for k in k_list:
            correct_count = 0
            class_correct = {1: 0, -1: 0}
            class_total = {1: 0, -1: 0}
            y_true = []
            y_pred = []

            for distances, ground_truth_label in all_distances:
                # Take the k nearest neighbors
                k_nearest_neighbors = distances[:k]
                # Determine the most common label
                labels_counts = {}
                for _, label in k_nearest_neighbors:
                    labels_counts[label] = labels_counts.get(label, 0) + 1
                predicted_label = max(labels_counts, key=labels_counts.get)

                y_true.append(ground_truth_label)
                y_pred.append(predicted_label)

                if predicted_label == ground_truth_label:
                    correct_count += 1
                    class_correct[ground_truth_label] += 1
                class_total[ground_truth_label] += 1

            # Calculate overall and by-class accuracies
            classification_accuracy = (correct_count / len(X_test)) * 100
            class_acc_dict[k].append(classification_accuracy)

            # Calculate F1 score with appropriate averaging
            if method == "kfold":
                f1 = f1_score(y_true, y_pred, average='macro') * 100
            else:  # stratk
                f1 = f1_score(y_true, y_pred, average='weighted') * 100
            f1_score_dict[k].append(f1)

            # Calculate and store by-class accuracies
            for class_label in [1, -1]:
                if class_total[class_label] > 0:
                    class_accuracy = (class_correct[class_label] /
                                      class_total[class_label]) * 100
                    by_class_acc_dict[k][class_label].append(class_accuracy)
                    print(
                        f"k={k}, Class {class_label} Accuracy: {class_accuracy}%, Fold {i}"
                    )

            print(
                f"k={k}, Overall Classification Accuracy: {classification_accuracy}%, Fold {i}"
            )
            average_type = "macro" if method == "kfold" else "weighted"
            print(f"k={k}, F1 Score ({average_type} avg): {f1:.2f}%, Fold {i}")

    end = time.time()
    print("Total runtime, seconds:", end - start)

    # Print accuracy stats for each k and print best overall k
    best_k = k_list[0]
    best_acc = 0
    best_f1 = 0
    k_avg_accuracies = {}
    k_avg_f1_scores = {}
    k_std_accuracies = {}
    k_std_f1_scores = {}

    average_type = "macro" if method == "kfold" else "weighted"

    for k in k_list:
        avg_acc = sum(class_acc_dict[k]) / len(class_acc_dict[k])
        avg_f1 = sum(f1_score_dict[k]) / len(f1_score_dict[k])

        # Calculate standard deviations
        std_acc = np.std(class_acc_dict[k])
        std_f1 = np.std(f1_score_dict[k])
        std_class_1 = np.std(by_class_acc_dict[k][1])
        std_class_neg1 = np.std(by_class_acc_dict[k][-1])

        k_avg_accuracies[k] = {
            'overall':
            avg_acc,
            'class_1':
            sum(by_class_acc_dict[k][1]) / len(by_class_acc_dict[k][1]),
            'class_-1':
            sum(by_class_acc_dict[k][-1]) / len(by_class_acc_dict[k][-1])
        }
        k_avg_f1_scores[k] = avg_f1
        k_std_accuracies[k] = {
            'overall': std_acc,
            'class_1': std_class_1,
            'class_-1': std_class_neg1
        }
        k_std_f1_scores[k] = std_f1

        print(f"\nk={k} Average Overall Accuracy: {avg_acc:.2f}%")
        print(f"k={k} Average F1 Score ({average_type} avg): {avg_f1:.2f}%")
        for class_label in [1, -1]:
            print(
                f"k={k} Average Class {class_label} Accuracy: {k_avg_accuracies[k][f'class_{class_label}']:.2f}%"
            )

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_k = k

        if avg_f1 > best_f1:
            best_f1 = avg_f1

    print(f"\nBest performance achieved with k={best_k}:")
    print(
        f"Overall Accuracy: {k_avg_accuracies[best_k]['overall']:.2f}% ± {k_std_accuracies[best_k]['overall']:.2f}%"
    )
    print(
        f"F1 Score ({average_type} avg): {k_avg_f1_scores[best_k]:.2f}% ± {k_std_f1_scores[best_k]:.2f}%"
    )
    print(
        f"Class 1 Accuracy: {k_avg_accuracies[best_k]['class_1']:.2f}% ± {k_std_accuracies[best_k]['class_1']:.2f}%"
    )
    print(
        f"Class -1 Accuracy: {k_avg_accuracies[best_k]['class_-1']:.2f}% ± {k_std_accuracies[best_k]['class_-1']:.2f}%"
    )

    # Prepare best k information to return
    best_k_info = {
        'best_k': best_k,
        'best_overall_accuracy': k_avg_accuracies[best_k]['overall'],
        'best_f1_score': k_avg_f1_scores[best_k],
        'best_class_1_accuracy': k_avg_accuracies[best_k]['class_1'],
        'best_class_minus1_accuracy': k_avg_accuracies[best_k]['class_-1']
    }

    return class_acc_dict, by_class_acc_dict, f1_score_dict, best_k_info
