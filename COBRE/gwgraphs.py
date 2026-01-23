import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import ot
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
Gromov-Wasserstein barycenter classification algorithm
"""
def gw_barycenter_class_alg(distance_matrices, labels, num_folds, num_nodes, seed=42, log=False):
    class_acc_list = []  # list to store classification accuracies from each fold
    class1_barys = []  # list to store class1 barycenters from each fold
    class2_barys = []  # list to store class2 barycenters from each fold

    skf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(skf.split(distance_matrices, labels)):
      print(f"Fold {i}:")
      if log is True:
          print(f"  Train: index={train_index}")
          print(f"  Test:  index={test_index}")
      X_train, X_test = [distance_matrices[idx] for idx in train_index
                         ], [distance_matrices[idx] for idx in test_index]
      y_train, y_test = [labels[idx] for idx in train_index
                         ], [labels[idx] for idx in test_index]
      X_train_class1 = [X_train[j] for j in range(len(X_train)) if y_train[j] == -1]
      X_train_class2 = [X_train[j] for j in range(len(X_train)) if y_train[j] == 1]
      # Compute the GW barycenter for each category
      barycenter_matrix_class1 = ot.gromov.gromov_barycenters(N=num_nodes,
                                                           Cs=X_train_class1,
                                                           loss_fun='square_loss',
                                                           random_state=seed)
      if log is True:
          print("GW barycenter for Class 1 graphs:", barycenter_matrix_class1, f"Fold {i}")
      class1_barys.append(barycenter_matrix_class1)
      print("GW barycenter for Class 1 graphs computed")
      barycenter_matrix_class2 = ot.gromov.gromov_barycenters(N=num_nodes,
                                                          Cs=X_train_class2,
                                                          loss_fun='square_loss',
                                                          random_state=seed)
      class2_barys.append(barycenter_matrix_class2)
      if log is True:
          print("GW barycenter for Class 2 graphs:", barycenter_matrix_class2, f"Fold {i}")
      print("GW barycenter for Class 2 graphs computed")

      gw, log_dists = ot.gromov.gromov_wasserstein(barycenter_matrix_class1,
                                                       barycenter_matrix_class2,
                                                       loss_fun='square_loss',
                                                       log=True)
      print("Distance between barycenters: ", log_dists['gw_dist'], f" Fold {i}")
      correct_count = 0
      correct_count_class1 = 0
      correct_count_class2 = 0
      total_count_class1 = 0
      total_count_class2 = 0

      predicted_labels = []
      true_labels = []
      total_samples = len(X_test)
      for idx, graph in enumerate(X_test):
        gw_class1, log_class1 = ot.gromov.gromov_wasserstein(graph,
                                                       barycenter_matrix_class1,
                                                       loss_fun='square_loss',
                                                       log=True)
        gw_class2, log_class2 = ot.gromov.gromov_wasserstein(graph,
                                                     barycenter_matrix_class2,
                                                     loss_fun='square_loss',
                                                     log=True)
        ground_truth_label = y_test[idx]
        true_labels.append(ground_truth_label)
        # Check if classification is correct
        predicted_label = -1 if log_class1['gw_dist'] < log_class2['gw_dist'] else 1
        predicted_labels.append(predicted_label)
        if ground_truth_label == -1:
          total_count_class1 += 1
          if predicted_label == ground_truth_label:
            correct_count_class1 += 1
        if ground_truth_label == 1:
          total_count_class2 += 1
          if predicted_label == ground_truth_label:
            correct_count_class2 += 1
        if predicted_label == ground_truth_label:
          correct_count += 1
      # Calculate classification accuracy
      classification_accuracy = (correct_count / total_samples) * 100
      class1_accuracy = (correct_count_class1 / total_count_class1) * 100
      class2_accuracy = (correct_count_class2 / total_count_class2) * 100
      print(f"Classification Accuracy: {classification_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 1: {class1_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 2: {class2_accuracy}%", f"Fold {i}")
      class_acc_list.append(classification_accuracy)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)
    return class_acc_list, class1_barys, class2_barys


def gw_barycenter_class_alg_stratkfold(distance_matrices, labels, num_folds, num_nodes, seed=42, log=False):
    class_acc_list = []  # list to store classification accuracies from each fold
    class1_barys = []  # list to store class1 barycenters from each fold
    class2_barys = []  # list to store class2 barycenters from each fold

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(skf.split(distance_matrices, labels)):
      print(f"Fold {i}:")
      if log is True:
          print(f"  Train: index={train_index}")
          print(f"  Test:  index={test_index}")
      X_train, X_test = [distance_matrices[idx] for idx in train_index
                         ], [distance_matrices[idx] for idx in test_index]
      y_train, y_test = [labels[idx] for idx in train_index
                         ], [labels[idx] for idx in test_index]
      X_train_class1 = [X_train[j] for j in range(len(X_train)) if y_train[j] == -1]
      X_train_class2 = [X_train[j] for j in range(len(X_train)) if y_train[j] == 1]
      # Compute the GW barycenter for each category
      barycenter_matrix_class1 = ot.gromov.gromov_barycenters(N=num_nodes,
                                                           Cs=X_train_class1,
                                                           loss_fun='square_loss',
                                                           random_state=seed)
      if log is True:
          print("GW barycenter for Class 1 graphs:", barycenter_matrix_class1, f"Fold {i}")
      class1_barys.append(barycenter_matrix_class1)
      print("GW barycenter for Class 1 graphs computed")
      barycenter_matrix_class2 = ot.gromov.gromov_barycenters(N=num_nodes,
                                                          Cs=X_train_class2,
                                                          loss_fun='square_loss',
                                                          random_state=seed)
      class2_barys.append(barycenter_matrix_class2)
      if log is True:
          print("GW barycenter for Class 2 graphs:", barycenter_matrix_class2, f"Fold {i}")
      print("GW barycenter for Class 2 graphs computed")

      gw, log_dists = ot.gromov.gromov_wasserstein(barycenter_matrix_class1,
                                                       barycenter_matrix_class2,
                                                       loss_fun='square_loss',
                                                       log=True)
      print("Distance between barycenters: ", log_dists['gw_dist'], f" Fold {i}")

      correct_count = 0
      correct_count_class1 = 0
      correct_count_class2 = 0
      total_count_class1 = 0
      total_count_class2 = 0

      predicted_labels = []
      true_labels = []
      total_samples = len(X_test)
      for idx, graph in enumerate(X_test):
        gw_class1, log_class1 = ot.gromov.gromov_wasserstein(graph,
                                                       barycenter_matrix_class1,
                                                       loss_fun='square_loss',
                                                       log=True)
        gw_class2, log_class2 = ot.gromov.gromov_wasserstein(graph,
                                                     barycenter_matrix_class2,
                                                     loss_fun='square_loss',
                                                     log=True)
        ground_truth_label = y_test[idx]
        true_labels.append(ground_truth_label)
        # Check if classification is correct
        predicted_label = -1 if log_class1['gw_dist'] < log_class2['gw_dist'] else 1
        predicted_labels.append(predicted_label)
        if ground_truth_label == -1:
          total_count_class1 += 1
          if predicted_label == ground_truth_label:
            correct_count_class1 += 1
        if ground_truth_label == 1:
          total_count_class2 += 1
          if predicted_label == ground_truth_label:
            correct_count_class2 += 1
        if predicted_label == ground_truth_label:
          correct_count += 1
      # Calculate classification accuracy
      classification_accuracy = (correct_count / total_samples) * 100
      class1_accuracy = (correct_count_class1 / total_count_class1) * 100
      class2_accuracy = (correct_count_class2 / total_count_class2) * 100
      print(f"Classification Accuracy: {classification_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 1: {class1_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 2: {class2_accuracy}%", f"Fold {i}")
      class_acc_list.append(classification_accuracy)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)
    return class_acc_list, class1_barys, class2_barys
    

def gw_barycentersubset_class_alg(distance_matrices,
                                  labels,
                                  num_folds,
                                  num_nodes,
                                  seed=42,
                                  log=False):
    class_acc_list = [
    ]  # list to store classification accuracies from each fold
    class1_barys = []  # list to store class1 barycenters from each fold
    class2_barys = []  # list to store class2 barycenters from each fold
    skf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(skf.split(distance_matrices, labels)):
        print(f"Fold {i}:")
        if log:
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
        X_train, X_test = [distance_matrices[idx] for idx in train_index
                           ], [distance_matrices[idx] for idx in test_index]
        y_train, y_test = [labels[idx] for idx in train_index
                           ], [labels[idx] for idx in test_index]

        X_train_class1 = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == -1
        ]
        X_train_class2 = [
            X_train[j] for j in range(len(X_train)) if y_train[j] == 1
        ]

        # Randomly select 20% of each category
        subset_class1 = random.sample(X_train_class1,
                                      k=max(1,
                                            len(X_train_class1) // 5))
        subset_class2 = random.sample(X_train_class2,
                                      k=max(1,
                                            len(X_train_class2) // 5))

        # Compute the GW barycenter for each category using the subset
        barycenter_matrix_class1 = ot.gromov.gromov_barycenters(
            N=num_nodes,
            Cs=subset_class1,
            loss_fun='square_loss',
            random_state=seed)
        if log:
            print(
                f"GW barycenter for Class 1 graphs (subset): {barycenter_matrix_class1}, Fold {i}"
            )
        class1_barys.append(barycenter_matrix_class1)
        print("GW barycenter for Class 1 graphs (subset) computed")

        barycenter_matrix_class2 = ot.gromov.gromov_barycenters(
            N=num_nodes,
            Cs=subset_class2,
            loss_fun='square_loss',
            random_state=seed)
        class2_barys.append(barycenter_matrix_class2)
        if log:
            print(
                f"GW barycenter for Class 2 graphs (subset): {barycenter_matrix_class2}, Fold {i}"
            )
        print("GW barycenter for Class 2 graphs (subset) computed")
        correct_count = 0
        correct_count_class1 = 0
        correct_count_class2 = 0
        total_count_class1 = 0
        total_count_class2 = 0
        predicted_labels = []
        true_labels = []
        total_samples = len(X_test)
        for idx, graph in enumerate(X_test):
            gw_class1, log_class1 = ot.gromov.gromov_wasserstein(
                graph,
                barycenter_matrix_class1,
                loss_fun='square_loss',
                log=True)
            gw_class2, log_class2 = ot.gromov.gromov_wasserstein(
                graph,
                barycenter_matrix_class2,
                loss_fun='square_loss',
                log=True)
            ground_truth_label = y_test[idx]
            true_labels.append(ground_truth_label)

            bary_gw, log_bary_gw = ot.gromov.gromov_wasserstein(
                barycenter_matrix_class1,
                barycenter_matrix_class2,
                loss_fun='square_loss',
                log=True)

            print("GW distance between test graph and Barycenter 1: ", {log_class1['gw_dist']})
            print("GW distance between test graph and Barycenter 2: ", {log_class2['gw_dist']})
            print("GW distance between the two barycenters: ", log_bary_gw['gw_dist'])

            # Check if classification is correct
            predicted_label = -1 if log_class1['gw_dist'] < log_class2[
                'gw_dist'] else 1
            predicted_labels.append(predicted_label)

            if ground_truth_label == -1:
                total_count_class1 += 1
                if predicted_label == ground_truth_label:
                    correct_count_class1 += 1
            if ground_truth_label == 1:
                total_count_class2 += 1
                if predicted_label == ground_truth_label:
                    correct_count_class2 += 1
            if predicted_label == ground_truth_label:
                correct_count += 1

        # Calculate classification accuracy
        classification_accuracy = (correct_count / total_samples) * 100
        class1_accuracy = (correct_count_class1 / total_count_class1) * 100
        class2_accuracy = (correct_count_class2 / total_count_class2) * 100

        print(f"Classification Accuracy: {classification_accuracy}%, Fold {i}")
        print(
            f"Classification Accuracy for Class 1: {class1_accuracy}%, Fold {i}"
        )
        print(
            f"Classification Accuracy for Class 2: {class2_accuracy}%, Fold {i}"
        )

        class_acc_list.append(classification_accuracy)

    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)
    return class_acc_list, class1_barys, class2_barys


def gw_barycenter_class_alg_var_nodes(distance_matrices, labels, num_folds, num_nodes_class1, num_nodes_class2, seed=42, log=False):
    class_acc_list = []  # list to store classification accuracies from each fold
    class1_barys = []  # list to store class1 barycenters from each fold
    class2_barys = []  # list to store class2 barycenters from each fold

    skf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(skf.split(distance_matrices, labels)):
      print(f"Fold {i}:")
      if log is True:
          print(f"  Train: index={train_index}")
          print(f"  Test:  index={test_index}")
      X_train, X_test = [distance_matrices[idx] for idx in train_index
                         ], [distance_matrices[idx] for idx in test_index]
      y_train, y_test = [labels[idx] for idx in train_index
                         ], [labels[idx] for idx in test_index]
      X_train_class1 = [X_train[j] for j in range(len(X_train)) if y_train[j] == -1]
      X_train_class2 = [X_train[j] for j in range(len(X_train)) if y_train[j] == 1]
      # Compute the GW barycenter for each category
      barycenter_matrix_class1 = ot.gromov.gromov_barycenters(N=num_nodes_class1,
                                                           Cs=X_train_class1,
                                                           loss_fun='square_loss',
                                                           random_state=seed)
      if log is True:
          print("GW barycenter for Class 1 graphs:", barycenter_matrix_class1, f"Fold {i}")
      class1_barys.append(barycenter_matrix_class1)
      print("GW barycenter for Class 1 graphs computed")
      barycenter_matrix_class2 = ot.gromov.gromov_barycenters(N=num_nodes_class2,
                                                          Cs=X_train_class2,
                                                          loss_fun='square_loss',
                                                          random_state=seed)
      class2_barys.append(barycenter_matrix_class2)
      if log is True:
          print("GW barycenter for Class 2 graphs:", barycenter_matrix_class2, f"Fold {i}")
      print("GW barycenter for Class 2 graphs computed")

      correct_count = 0
      correct_count_class1 = 0
      correct_count_class2 = 0
      total_count_class1 = 0
      total_count_class2 = 0

      predicted_labels = []
      true_labels = []
      total_samples = len(X_test)
      for idx, graph in enumerate(X_test):
        gw_class1, log_class1 = ot.gromov.gromov_wasserstein(graph,
                                                       barycenter_matrix_class1,
                                                       loss_fun='square_loss',
                                                       log=True)
        gw_class2, log_class2 = ot.gromov.gromov_wasserstein(graph,
                                                     barycenter_matrix_class2,
                                                     loss_fun='square_loss',
                                                     log=True)
        ground_truth_label = y_test[idx]
        true_labels.append(ground_truth_label)
        # Check if classification is correct
        predicted_label = -1 if log_class1['gw_dist'] < log_class2['gw_dist'] else 1
        predicted_labels.append(predicted_label)
        if ground_truth_label == -1:
          total_count_class1 += 1
          if predicted_label == ground_truth_label:
            correct_count_class1 += 1
        if ground_truth_label == 1:
          total_count_class2 += 1
          if predicted_label == ground_truth_label:
            correct_count_class2 += 1
        if predicted_label == ground_truth_label:
          correct_count += 1
      # Calculate classification accuracy
      classification_accuracy = (correct_count / total_samples) * 100
      class1_accuracy = (correct_count_class1 / total_count_class1) * 100
      class2_accuracy = (correct_count_class2 / total_count_class2) * 100
      print(f"Classification Accuracy: {classification_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 1: {class1_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 2: {class2_accuracy}%", f"Fold {i}")
      class_acc_list.append(classification_accuracy)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)
    return class_acc_list, class1_barys, class2_barys


def gw_barycenter_class_alg_var_nodes_stratkfold(distance_matrices, labels, num_folds, num_nodes_class1, num_nodes_class2, seed=42, log=False):
    class_acc_list = []  # list to store classification accuracies from each fold
    class1_barys = []  # list to store class1 barycenters from each fold
    class2_barys = []  # list to store class2 barycenters from each fold

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    start = time.time()
    for i, (train_index,
            test_index) in enumerate(skf.split(distance_matrices, labels)):
      print(f"Fold {i}:")
      if log is True:
          print(f"  Train: index={train_index}")
          print(f"  Test:  index={test_index}")
      X_train, X_test = [distance_matrices[idx] for idx in train_index
                         ], [distance_matrices[idx] for idx in test_index]
      y_train, y_test = [labels[idx] for idx in train_index
                         ], [labels[idx] for idx in test_index]
      X_train_class1 = [X_train[j] for j in range(len(X_train)) if y_train[j] == -1]
      X_train_class2 = [X_train[j] for j in range(len(X_train)) if y_train[j] == 1]
      # Compute the GW barycenter for each category
      barycenter_matrix_class1 = ot.gromov.gromov_barycenters(N=num_nodes_class1,
                                                           Cs=X_train_class1,
                                                           loss_fun='square_loss',
                                                           random_state=seed)
      if log is True:
          print("GW barycenter for Class 1 graphs:", barycenter_matrix_class1, f"Fold {i}")
      class1_barys.append(barycenter_matrix_class1)
      print("GW barycenter for Class 1 graphs computed")
      barycenter_matrix_class2 = ot.gromov.gromov_barycenters(N=num_nodes_class2,
                                                          Cs=X_train_class2,
                                                          loss_fun='square_loss',
                                                          random_state=seed)
      class2_barys.append(barycenter_matrix_class2)
      if log is True:
          print("GW barycenter for Class 2 graphs:", barycenter_matrix_class2, f"Fold {i}")
      print("GW barycenter for Class 2 graphs computed")

      correct_count = 0
      correct_count_class1 = 0
      correct_count_class2 = 0
      total_count_class1 = 0
      total_count_class2 = 0

      predicted_labels = []
      true_labels = []
      total_samples = len(X_test)
      for idx, graph in enumerate(X_test):
        gw_class1, log_class1 = ot.gromov.gromov_wasserstein(graph,
                                                       barycenter_matrix_class1,
                                                       loss_fun='square_loss',
                                                       log=True)
        gw_class2, log_class2 = ot.gromov.gromov_wasserstein(graph,
                                                     barycenter_matrix_class2,
                                                     loss_fun='square_loss',
                                                     log=True)
        ground_truth_label = y_test[idx]
        true_labels.append(ground_truth_label)
        # Check if classification is correct
        predicted_label = -1 if log_class1['gw_dist'] < log_class2['gw_dist'] else 1
        predicted_labels.append(predicted_label)
        if ground_truth_label == -1:
          total_count_class1 += 1
          if predicted_label == ground_truth_label:
            correct_count_class1 += 1
        if ground_truth_label == 1:
          total_count_class2 += 1
          if predicted_label == ground_truth_label:
            correct_count_class2 += 1
        if predicted_label == ground_truth_label:
          correct_count += 1
      # Calculate classification accuracy
      classification_accuracy = (correct_count / total_samples) * 100
      class1_accuracy = (correct_count_class1 / total_count_class1) * 100
      class2_accuracy = (correct_count_class2 / total_count_class2) * 100
      print(f"Classification Accuracy: {classification_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 1: {class1_accuracy}%", f"Fold {i}")
      print(f"Classification Accuracy for Class 2: {class2_accuracy}%", f"Fold {i}")
      class_acc_list.append(classification_accuracy)
    end = time.time()
    print("Total runtime, seconds: ")
    print(end - start)
    return class_acc_list, class1_barys, class2_barys
