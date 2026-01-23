
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GCNResultsVisualizer:
    def __init__(self, study=None, all_results=None, best_params=None):
        self.study = study
        self.all_results = all_results
        self.best_params = best_params
        
    def plot_optimization_history(self, save_path="optimization_history.png"):
        """Plot the optimization history showing improvement over trials"""
        if self.study is None:
            print("No study provided for optimization history")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trial history
        trials = self.study.trials
        trial_numbers = [t.number for t in trials if t.value is not None]
        trial_values = [t.value for t in trials if t.value is not None]
        
        ax1.plot(trial_numbers, trial_values, 'b-', alpha=0.6, linewidth=1)
        ax1.scatter(trial_numbers, trial_values, c='blue', alpha=0.8, s=30)
        
        # Running best
        running_best = []
        best_so_far = -np.inf
        for val in trial_values:
            if val > best_so_far:
                best_so_far = val
            running_best.append(best_so_far)
        
        ax1.plot(trial_numbers, running_best, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Parameter importance
        importance = self.study.trials_dataframe()
        if len(importance) > 5:  # Only plot if we have enough trials
            param_importance = {}
            for param in self.best_params.keys():
                if f'params_{param}' in importance.columns:
                    correlation = abs(importance[f'params_{param}'].corr(importance['value']))
                    param_importance[param] = correlation
            
            params = list(param_importance.keys())
            importances = list(param_importance.values())
            
            ax2.barh(params, importances)
            ax2.set_xlabel('Parameter Importance (Correlation)')
            ax2.set_title('Hyperparameter Importance')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hyperparameter_relationships(self, save_path="hyperparameter_relationships.png"):
        """Plot relationships between hyperparameters and performance"""
        if self.study is None:
            return
            
        df = self.study.trials_dataframe()
        
        # Create parameter plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Learning rate vs accuracy
        if 'params_learning_rate' in df.columns:
            axes[0].scatter(df['params_learning_rate'], df['value'], alpha=0.7, c='blue')
            axes[0].set_xlabel('Learning Rate')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_xscale('log')
            axes[0].set_title('Learning Rate vs Accuracy')
            axes[0].grid(True, alpha=0.3)
        
        # Epochs vs accuracy
        if 'params_num_epochs' in df.columns:
            axes[1].scatter(df['params_num_epochs'], df['value'], alpha=0.7, c='green')
            axes[1].set_xlabel('Number of Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Epochs vs Accuracy')
            axes[1].grid(True, alpha=0.3)
        
        # Batch size vs accuracy
        if 'params_batch_size' in df.columns:
            axes[2].scatter(df['params_batch_size'], df['value'], alpha=0.7, c='red')
            axes[2].set_xlabel('Batch Size')
            axes[2].set_ylabel('Accuracy')
            axes[2].set_title('Batch Size vs Accuracy')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cross_validation_results(self, save_path="cv_results.png"):
        """Plot cross-validation results across folds"""
        if self.all_results is None:
            return
            
        # Extract metrics
        metrics = ['overall_accuracy', 'class_1_accuracy', 'class_-1_accuracy', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [r[metric] for r in self.all_results]
            folds = [r['fold'] for r in self.all_results]
            
            # Box plot
            axes[i].boxplot([values], labels=[metric.replace('_', ' ').title()])
            
            # Individual points
            axes[i].scatter([1] * len(values), values, alpha=0.7, s=50)
            
            # Add fold labels
            for j, (fold, val) in enumerate(zip(folds, values)):
                axes[i].annotate(f'F{fold}', (1, val), xytext=(5, 0), 
                               textcoords='offset points', fontsize=8)
            
            axes[i].set_ylabel('Score')
            axes[i].set_title(f'{metric.replace("_", " ").title()} Across Folds')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = np.mean(values)
            axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                           label=f'Mean: {mean_val:.4f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        """Plot confusion matrix for the final model"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class -1', 'Class +1'],
                   yticklabels=['Class -1', 'Class +1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Best Model')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, save_path="roc_curve.png"):
        """Plot ROC curve for binary classification"""
        # Convert labels to binary (0, 1)
        y_true_binary = (y_true + 1) // 2
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores[:, 1])  # Use probability of positive class
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Best Model')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_graphs(self, graphs, labels, num_samples=6, save_path="sample_graphs.png"):
        """Visualize sample graphs from the dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Sample graphs
        indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        
        for i, idx in enumerate(indices):
            if i >= num_samples:
                break
                
            graph = graphs[idx]
            label = labels[idx]
            
            # Create layout
            pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # Draw graph
            nx.draw(graph, pos, ax=axes[i], 
                   node_color='lightblue' if label == 1 else 'lightcoral',
                   node_size=100, 
                   with_labels=False,
                   edge_color='gray',
                   alpha=0.8)
            
            axes[i].set_title(f'Graph {idx} (Label: {label})')
            axes[i].set_aspect('equal')
        
        # Remove empty subplots
        for i in range(len(indices), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, training_history, save_path="learning_curves.png"):
        """Plot training and validation curves if available"""
        if not training_history:
            print("No training history provided")
            return
            
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in training_history:
            ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        if 'train_acc' in training_history:
            ax2.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy')
        if 'val_acc' in training_history:
            ax2.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training History - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard using Plotly"""
        if self.study is None:
            return
            
        df = self.study.trials_dataframe()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization History', 'Parameter vs Accuracy',
                          'Parameter Importance', 'Cross-validation Results'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Optimization history
        fig.add_trace(
            go.Scatter(x=df['number'], y=df['value'], mode='lines+markers',
                      name='Trial Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Parameter relationship (learning rate)
        if 'params_learning_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['params_learning_rate'], y=df['value'], 
                          mode='markers', name='Learning Rate vs Accuracy',
                          marker=dict(color='green')),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, 
                         title_text="GCN Optimization Results Dashboard")
        
        return fig
    
    def generate_all_visualizations(self, graphs_list=None, labels_list=None, 
                                  y_true=None, y_pred=None, y_scores=None):
        """Generate all available visualizations"""
        print("Generating comprehensive visualization suite...")
        
        # Core optimization plots
        if self.study is not None:
            self.plot_optimization_history()
            self.plot_hyperparameter_relationships()
        
        # Cross-validation results
        if self.all_results is not None:
            self.plot_cross_validation_results()
        
        # Model performance plots
        if y_true is not None and y_pred is not None:
            self.plot_confusion_matrix(y_true, y_pred)
            
        if y_true is not None and y_scores is not None:
            self.plot_roc_curve(y_true, y_scores)
        
        # Graph visualization
        if graphs_list is not None and labels_list is not None:
            self.plot_sample_graphs(graphs_list, labels_list)
        
        print("All visualizations generated successfully!")

# Function to extract model predictions with probabilities for ROC curve
def get_model_probabilities(model, test_loader, device):
    """Get model predictions with probabilities for ROC analysis"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(out, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)
