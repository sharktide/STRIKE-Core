"""
Script to generate confusion matrices for all disaster prediction models
with and without TrustNet, saving visualizations to the matrices folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import tensorflow as tf

# Enable unsafe deserialization for lambda layers
tf.keras.config.enable_unsafe_deserialization()

from objects import *
from load import *

os.makedirs("matrices", exist_ok=True)
accel = "/cpu:0"  # Default to CPU
if (tf.config.list_physical_devices('GPU')):
    print("GPU detected. Using GPU for computations.")
    accel = "/gpu:0"

MODELS_CONFIG = {
    'Fire': {
        'base_model': FireNet,
        'trust_model': FireTrustNet,
        'scaler': FireScaler,
        'dataset': 'dataset/final_fire_dataset2.csv',
        'label_col': 'fire_occurred',
        'features': ['temperature', 'humidity', 'wind_speed', 'vegetation_index', 'elevation']
    },
    'FlashFlood': {
        'base_model': FlashFloodNet,
        'trust_model': FlashFloodTrustNet,
        'scaler': FlashFloodScaler,
        'dataset': 'dataset/flash_flood_data.csv',
        'label_col': 'flash_binary',
        'features': ['rainfall_intensity', 'slope', 'drainage_density', 'soil_saturation', 'convergence_index']
    },
    'Flood': {
        'base_model': FloodNet,
        'trust_model': FloodTrustNet,
        'scaler': FloodScaler,
        'dataset': 'dataset/sampled_flood_data.csv',
        'label_col': 'flood_binary',
        'features': ['rainfall', 'water_level', 'elevation', 'slope', 'distance_from_river']
    },
    'PVFlood': {
        'base_model': PV_FloodNet,
        'trust_model': PV_FloodTrustNet,
        'scaler': PV_FloodScaler,
        'dataset': 'dataset/pluvial_flood_data_balanced.csv',
        'label_col': 'pluvial_binary',
        'features': ['rainfall_intensity', 'impervious_ratio', 'drainage_density', 'urbanization_index', 'convergence_index']
    },
    'Quake': {
        'base_model': QuakeNet,
        'trust_model': QuakeTrustNet,
        'scaler': QuakeTrustScaler,
        'dataset': 'dataset/earthquake_data.csv',
        'label_col': 'quake_binary',
        'features': ['seismic_moment_rate', 'surface_displacement_rate', 'coulomb_stress_change', 'average_focal_depth', 'fault_slip_rate']
    },
    'Hurricane': {
        'base_model': HurricaneNet,
        'trust_model': HurricaneTrustNet,
        'scaler': HurricaneTrustScaler,
        'dataset': 'dataset/hurricane_data.csv',
        'label_col': 'hurricane_binary',
        'features': ['sea_surface_temperature', 'ocean_heat_content', 'mid_level_humidity', 'vertical_wind_shear', 'potential_vorticity']
    },
    'Tornado': {
        'base_model': TornadoNet,
        'trust_model': TornadoTrustNet,
        'scaler': TornadoTrustScaler,
        'dataset': 'dataset/tornado_data.csv',
        'label_col': 'tornado_binary',
        'features': ['storm_relative_helicity', 'CAPE', 'lifted_condensation_level', 'bulk_wind_shear', 'significant_tornado_param']
    }
}


def generate_predictions(model, X, trust_model=None, scaler=None, use_trust=False):
    """
    Generate predictions from a model, optionally using TrustNet.
    
    Args:
        model: Base prediction model
        X: Input features
        trust_model: Optional TrustNet model
        scaler: Optional scaler for trust model input
        use_trust: Whether to apply TrustNet weighting
    
    Returns:
        Array of binary predictions (0 or 1)
    """
    with tf.device(accel):
        base_predictions = model.predict(X, verbose=0)
        
        if use_trust and trust_model is not None and scaler is not None:
            scaled_X = scaler.transform(X)
            trust_scores = trust_model.predict(scaled_X, verbose=0)
            # Combine base predictions with trust scores
            combined = base_predictions * trust_scores
            predictions = (combined > 0.5).astype(int).flatten()
        else:
            predictions = (base_predictions > 0.5).astype(int).flatten()
        
        return predictions


def create_confusion_matrix_plot(y_true, y_pred, title, save_path):
    """Create and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return cm


def print_metrics(y_true, y_pred, cm):
    """Print classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   zero_division=0)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'report': report
    }


def process_model(model_name, config):
    """Process a single model and generate confusion matrices."""
    print(f"\n{'='*60}")
    print(f"Processing {model_name}...")
    print(f"{'='*60}")
    
    # Load dataset
    df = pd.read_csv(config['dataset'])
    X = df[config['features']]
    y = df[config['label_col']].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Generate predictions without TrustNet
    print(f"\nGenerating predictions without TrustNet...")
    y_pred_base = generate_predictions(config['base_model'], X, use_trust=False)
    
    # Generate predictions with TrustNet
    print(f"Generating predictions with TrustNet...")
    y_pred_trust = generate_predictions(
        config['base_model'], X,
        trust_model=config['trust_model'],
        scaler=config['scaler'],
        use_trust=True
    )
    
    # Create confusion matrices without TrustNet
    title_base = f"{model_name} - Without TrustNet"
    path_base = f"matrices/{model_name}_without_trustnet.png"
    cm_base = create_confusion_matrix_plot(y, y_pred_base, title_base, path_base)
    print(f"✓ Saved: {path_base}")
    
    # Create confusion matrices with TrustNet
    title_trust = f"{model_name} - With TrustNet"
    path_trust = f"matrices/{model_name}_with_trustnet.png"
    cm_trust = create_confusion_matrix_plot(y, y_pred_trust, title_trust, path_trust)
    print(f"✓ Saved: {path_trust}")
    
    # Print metrics for both versions
    print(f"\n--- Metrics Without TrustNet ---")
    metrics_base = print_metrics(y, y_pred_base, cm_base)
    print(f"Accuracy:    {metrics_base['accuracy']:.4f}")
    print(f"Sensitivity: {metrics_base['sensitivity']:.4f}")
    print(f"Specificity: {metrics_base['specificity']:.4f}")
    print(f"Precision:   {metrics_base['precision']:.4f}")
    
    print(f"\n--- Metrics With TrustNet ---")
    metrics_trust = print_metrics(y, y_pred_trust, cm_trust)
    print(f"Accuracy:    {metrics_trust['accuracy']:.4f}")
    print(f"Sensitivity: {metrics_trust['sensitivity']:.4f}")
    print(f"Specificity: {metrics_trust['specificity']:.4f}")
    print(f"Precision:   {metrics_trust['precision']:.4f}")
    
    # Calculate improvement
    acc_improvement = (metrics_trust['accuracy'] - metrics_base['accuracy']) * 100
    print(f"\nAccuracy Improvement: {acc_improvement:+.2f}%")
    
    return {
        'model': model_name,
        'metrics_base': metrics_base,
        'metrics_trust': metrics_trust,
        'improvement': acc_improvement
    }


def main():
    """Main execution function."""
    print("="*60)
    print("Confusion Matrix Generation for All Models")
    print("="*60)
    
    results = []
    
    # Process each model
    for model_name, config in MODELS_CONFIG.items():
        try:
            result = process_model(model_name, config)
            results.append(result)
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary report
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}\n")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model'],
            'Accuracy (Base)': f"{result['metrics_base']['accuracy']:.4f}",
            'Accuracy (Trust)': f"{result['metrics_trust']['accuracy']:.4f}",
            'Improvement': f"{result['improvement']:+.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv('matrices/confusion_matrices_summary.csv', index=False)
    print(f"\n✓ Summary saved to: matrices/confusion_matrices_summary.csv")
    
    print(f"\n✓ All confusion matrices saved to 'matrices' folder")


if __name__ == "__main__":
    main()
