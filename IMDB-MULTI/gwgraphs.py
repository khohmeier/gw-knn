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
GW-kNN and fGW-kNN algorithms
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
