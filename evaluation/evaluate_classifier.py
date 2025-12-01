import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score, 
    precision_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path
from collections import defaultdict

def load_json_file(filepath):
    """Load JSON file and return data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_labels(data):
    """Extract num_vertices from data as labels."""
    return [item['num_vertices'] for item in data]

def calculate_metrics(y_true, y_pred, classes):
    """Calculate accuracy, precision, recall, and F1 per class."""
    metrics = {
        'overall': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        },
        'per_class': {}
    }
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    
    for i, cls in enumerate(classes):
        metrics['per_class'][f'class_{cls}'] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i])
        }
    
    return metrics

def print_confusion_matrix(y_true, y_pred, classes, model_name):
    """Print confusion matrix to terminal in a formatted way."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - CONFUSION MATRIX")
    print('='*70)
    print("\nRows = True Label, Columns = Predicted Label\n")
    
    # Determine column width based on maximum value
    max_val = cm.max()
    col_width = max(len(str(max_val)), len(str(max(classes)))) + 2
    
    # Print header
    print(" " * 8, end="")
    for cls in classes:
        print(f"{cls:>{col_width}}", end="")
    print()
    
    print(" " * 8 + "-" * (col_width * len(classes)))
    
    # Print rows
    for i, cls in enumerate(classes):
        print(f"{cls:>6} |", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>{col_width}}", end="")
        print()
    
    print()
    
    # Print totals
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    
    print(f"Row Totals (True class counts):")
    for i, cls in enumerate(classes):
        print(f"  Class {cls}: {row_sums[i]}")
    
    print(f"\nColumn Totals (Predicted class counts):")
    for i, cls in enumerate(classes):
        print(f"  Class {cls}: {col_sums[i]}")
    
    # Calculate diagonal accuracy
    diagonal_sum = np.trace(cm)
    total = cm.sum()
    print(f"\nCorrect predictions (diagonal): {diagonal_sum} / {total} = {diagonal_sum/total:.6f}")

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label (num_vertices)', fontsize=12)
    plt.xlabel('Predicted Label (num_vertices)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm.tolist()

def plot_metrics_comparison(baseline_metrics, classifier_metrics, classes, output_path):
    """Create comparison bar plots for metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_names = ['precision', 'recall', 'f1']
    
    # Prepare data for plotting
    x = np.arange(len(classes))
    width = 0.35
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        baseline_values = [baseline_metrics['per_class'][f'class_{cls}'][metric] for cls in classes]
        classifier_values = [classifier_metrics['per_class'][f'class_{cls}'][metric] for cls in classes]
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, classifier_values, width, label='Classifier', alpha=0.8)
        
        ax.set_xlabel('Number of Vertices', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.capitalize()} per Class', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # Overall accuracy comparison
    ax = axes[1, 1]
    models = ['Baseline', 'Classifier']
    accuracies = [
        baseline_metrics['overall']['accuracy'],
        classifier_metrics['overall']['accuracy']
    ]
    
    bars = ax.bar(models, accuracies, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Overall Accuracy Comparison', fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_classification_report(y_true, y_pred, model_name):
    """Print sklearn-style classification report with 6 decimal places."""
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - CLASSIFICATION REPORT")
    print('='*70)
    
    # Get all unique classes from both y_true and y_pred
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    
    # Print with 6 decimal places
    report = classification_report(y_true, y_pred, labels=all_classes, zero_division=0, digits=6)
    print(report)
    
    # Show which classes appear in predictions
    pred_classes = set(y_pred)
    true_classes = set(y_true)
    
    if pred_classes != true_classes:
        print(f"\nNote: Ground truth contains classes: {sorted(true_classes)}")
        print(f"      {model_name} predicted classes: {sorted(pred_classes)}")
        
        only_in_pred = pred_classes - true_classes
        if only_in_pred:
            print(f"      Classes predicted but not in ground truth: {sorted(only_in_pred)}")

def main():
    # File paths
    ground_truth_file = 'output/ground_truth_test_set.json'
    baseline_file = 'output/baseline_test_set.json'
    predictions_classifier_file = 'output/predictions_with_classifier.json'
    
    # Output directory
    output_dir = Path('output/evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data files...")
    ground_truth = load_json_file(ground_truth_file)
    baseline_predictions = load_json_file(baseline_file)
    classifier_predictions = load_json_file(predictions_classifier_file)
    
    # Extract labels
    y_true = extract_labels(ground_truth)
    y_baseline = extract_labels(baseline_predictions)
    y_classifier = extract_labels(classifier_predictions)
    
    # Get unique classes from ground truth (for consistent evaluation)
    classes_true = sorted(list(set(y_true)))
    
    # Get all unique classes for baseline (union of true and predicted)
    classes_baseline_all = sorted(list(set(y_true) | set(y_baseline)))
    
    # Get all unique classes for classifier (union of true and predicted)
    classes_classifier_all = sorted(list(set(y_true) | set(y_classifier)))
    
    print(f"\nGround truth classes (num_vertices): {classes_true}")
    print(f"Baseline all classes (true + predicted): {classes_baseline_all}")
    print(f"Classifier all classes (true + predicted): {classes_classifier_all}")
    
    # Print classification reports and confusion matrices (sklearn style)
    print_classification_report(y_true, y_baseline, "Baseline (Non-ML)")
    print_confusion_matrix(y_true, y_baseline, classes_baseline_all, "Baseline (Non-ML)")
    
    print_classification_report(y_true, y_classifier, "Classifier")
    print_confusion_matrix(y_true, y_classifier, classes_classifier_all, "Classifier")
    
    # Calculate metrics for comparison (using only ground truth classes for fair comparison)
    print("\n" + "="*70)
    print("CALCULATING METRICS FOR GROUND TRUTH CLASSES")
    print("="*70)
    baseline_metrics = calculate_metrics(y_true, y_baseline, classes_true)
    classifier_metrics = calculate_metrics(y_true, y_classifier, classes_true)
    
    # Generate confusion matrices (for plots, use the full class sets)
    print("\nGenerating confusion matrices...")
    baseline_cm = plot_confusion_matrix(
        y_true, y_baseline, classes_baseline_all,
        'Confusion Matrix - Baseline (Non-ML)',
        output_dir / 'confusion_matrix_baseline.png'
    )
    
    classifier_cm = plot_confusion_matrix(
        y_true, y_classifier, classes_classifier_all,
        'Confusion Matrix - Classifier',
        output_dir / 'confusion_matrix_classifier.png'
    )
    
    # Generate metrics comparison plot (only for ground truth classes)
    print("Generating metrics comparison plot...")
    plot_metrics_comparison(
        baseline_metrics, classifier_metrics, classes_true,
        output_dir / 'classifier_metrics_comparison.png'
    )
    
    # Prepare final results
    results = {
        'classes_ground_truth': classes_true,
        'classes_baseline_all': classes_baseline_all,
        'classes_classifier_all': classes_classifier_all,
        'baseline': {
            'metrics': baseline_metrics,
            'confusion_matrix': baseline_cm,
            'confusion_matrix_classes': classes_baseline_all
        },
        'classifier': {
            'metrics': classifier_metrics,
            'confusion_matrix': classifier_cm,
            'confusion_matrix_classes': classes_classifier_all
        },
        'comparison': {
            'accuracy_improvement': float(
                classifier_metrics['overall']['accuracy'] - 
                baseline_metrics['overall']['accuracy']
            ),
            'f1_macro_improvement': float(
                classifier_metrics['overall']['f1_macro'] - 
                baseline_metrics['overall']['f1_macro']
            )
        }
    }
    
    # Save results to JSON
    print("\nSaving results to JSON...")
    with open(output_dir / 'classifier_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print("\nBASELINE (Non-ML):")
    print(f"  Accuracy: {baseline_metrics['overall']['accuracy']:.6f}")
    print(f"  Macro F1: {baseline_metrics['overall']['f1_macro']:.6f}")
    print(f"  Macro Precision: {baseline_metrics['overall']['precision_macro']:.6f}")
    print(f"  Macro Recall: {baseline_metrics['overall']['recall_macro']:.6f}")
    
    print("\nCLASSIFIER:")
    print(f"  Accuracy: {classifier_metrics['overall']['accuracy']:.6f}")
    print(f"  Macro F1: {classifier_metrics['overall']['f1_macro']:.6f}")
    print(f"  Macro Precision: {classifier_metrics['overall']['precision_macro']:.6f}")
    print(f"  Macro Recall: {classifier_metrics['overall']['recall_macro']:.6f}")
    
    print("\nIMPROVEMENT:")
    print(f"  Accuracy: {results['comparison']['accuracy_improvement']:+.6f}")
    print(f"  Macro F1: {results['comparison']['f1_macro_improvement']:+.6f}")
    
    print("\nPER-CLASS METRICS (CLASSIFIER):")
    for cls in classes_true:
        metrics = classifier_metrics['per_class'][f'class_{cls}']
        print(f"\n  Class {cls} vertices:")
        print(f"    Precision: {metrics['precision']:.6f}")
        print(f"    Recall: {metrics['recall']:.6f}")
        print(f"    F1: {metrics['f1']:.6f}")
    
    print("\n" + "="*70)
    print(f"\nResults saved to '{output_dir}/' directory:")
    print(f"  - classifier_evaluation_results.json")
    print(f"  - confusion_matrix_baseline.png")
    print(f"  - confusion_matrix_classifier.png")
    print(f"  - classifier_metrics_comparison.png")
    print("="*70)

if __name__ == "__main__":
    main()
