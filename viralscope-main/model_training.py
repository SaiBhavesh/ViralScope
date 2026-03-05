import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training and evaluation for YouTube trend prediction.
    
    Trains multiple classifiers and evaluates their performance
    on the stratified sample with engineered features.
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.evaluation_results = []
        self.best_params = {}
    
    def get_model_configs(self):
        """
        Get model configurations with base estimators and hyperparameter grids.
        
        Returns:
            Dictionary mapping model names to tuples of (estimator, param_grid)
        """
        configs = {
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=200, 
                    random_state=self.random_state, 
                    n_jobs=self.n_jobs
                ),
                {
                    'n_estimators': [200, 300],
                    'max_depth': [10, 20],
                    'min_samples_leaf': [1, 3],
                    'class_weight': ['balanced', None]
                }
            ),
            "DecisionTree": (
                DecisionTreeClassifier(
                    random_state=self.random_state
                ),
                {
                    'max_depth': [10, 20],
                    'min_samples_leaf': [3, 5],
                    'criterion': ['gini', 'entropy'],
                    'class_weight': ['balanced', None]
                }
            ),
            "LinearSVC": (
                LinearSVC(
                    random_state=self.random_state
                ),
                {
                    'C': [0.01, 0.1],
                    'class_weight': ['balanced', None],
                    'max_iter': [1000, 2000]
                }
            ),
            "KNN": (
                KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=self.n_jobs
                ),
                {
                    'n_neighbors': [7, 9],
                    #'weights': ['uniform', 'distance'],
                    'p': [2]
                }
            ),
            "MLP": (
                MLPClassifier(
                    hidden_layer_sizes=(128, 64), 
                    max_iter=400, 
                    random_state=self.random_state
                ),
                {
                    'hidden_layer_sizes': [(128, 64), (256, 128, 32)],
                    'learning_rate_init': [0.02, 0.05],
                    'activation': ['relu', 'logistic'],
                    'max_iter': [100, 200]
                }
            ),
            "LogisticRegression": (
                LogisticRegression(
                    random_state=self.random_state,
                    max_iter=200
                ),
                {
                    'C': [0.01, 0.1],
                    'class_weight': ['balanced', None],
                    'max_iter': [100, 200]
                }
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    random_state=self.random_state
                ),
                {
                    'n_estimators': [200, 300],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            )
        }
        return configs
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance and print metrics.
        """
        print(f"\n{'='*70}")
        print(f"{model_name.upper()} EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Handle probability predictions (not all models support predict_proba)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except AttributeError:
            # For models like LinearSVC that don't have predict_proba
            if hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred_proba = None
                roc_auc = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        
        # Print metrics
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store metrics for CSV export
        metrics_dict = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc if roc_auc is not None else 'N/A'
        }
        self.evaluation_results.append(metrics_dict)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def plot_feature_importance(self, model, feature_names, model_name, output_path, top_n=20):
        """
        Plot and save feature importance.
        """
        importances = model.feature_importances_
        
        # Limit top_n to actual number of features
        actual_top_n = min(top_n, len(feature_names))
        indices = np.argsort(importances)[::-1][:actual_top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {actual_top_n} Features - {model_name}', fontsize=14, fontweight='bold')
        plt.bar(range(len(indices)), importances[indices], color='mediumseagreen', edgecolor='black', align='center')
        
        # Get feature names using the indices
        feature_labels = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
        plt.xticks(range(len(indices)), feature_labels, rotation=45, ha='right')
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved: {output_path}")
    
    def plot_decision_tree(self, model, feature_names, model_name, output_path, max_depth=3):
        """
        Plot and save decision tree visualization.
        
        Args:
            model: Trained DecisionTreeClassifier
            feature_names: List of feature names
            model_name: Name of the model
            output_path: Path to save the tree plot
            max_depth: Maximum depth to display (to avoid overcrowding)
        """
        plt.figure(figsize=(20, 12))
        plot_tree(model, 
                  feature_names=feature_names,
                  class_names=['Not Successful', 'Successful'],
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  max_depth=max_depth)
        plt.title(f'Decision Tree Visualization - {model_name} (max_depth={max_depth} shown)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Decision tree visualization saved: {output_path}")
    
    def plot_confusion_matrix(self, cm, model_name, output_path):
        """
        Plot and save confusion matrix heatmap.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Not Successful', 'Successful'],
                    yticklabels=['Not Successful', 'Successful'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved: {output_path}")
    
    def save_models(self, output_dir='Models'):
        """
        Save trained models to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved: {model_path}")
    
    def save_evaluation_metrics(self, output_dir='Models'):
        """
        Save evaluation metrics to CSV file.
        """
        if not self.evaluation_results:
            print("No evaluation results to save.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
        
        metrics_df = pd.DataFrame(self.evaluation_results)
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"\nEvaluation metrics saved: {metrics_path}")
        print("\nMetrics Summary:")
        print(metrics_df.to_string(index=False))
        
        # Save best hyperparameters to separate file
        if self.best_params:
            params_path = os.path.join(output_dir, 'best_hyperparameters.txt')
            with open(params_path, 'w') as f:
                f.write("Best Hyperparameters from GridSearchCV\n")
                f.write("="*70 + "\n\n")
                for model_name, params in self.best_params.items():
                    f.write(f"{model_name}:\n")
                    for param, value in params.items():
                        f.write(f"  {param}: {value}\n")
                    f.write("\n")
            print(f"Best hyperparameters saved: {params_path}")
    
    def run_training_pipeline(self, X_train, X_test, y_train, y_test, feature_names, output_dir='Models'):
        """
        Execute complete model training and evaluation pipeline.
        
        Args:
            X_train: Training feature matrix (pre-scaled)
            X_test: Testing feature matrix (pre-scaled)
            y_train: Training labels
            y_test: Testing labels
            feature_names: List of feature column names
            output_dir: Directory to save models and plots
            
        Returns:
            Dictionary with trained models and evaluation results
        """
        print("="*70)
        print("MODEL TRAINING PIPELINE")
        print("="*70)
        
        print(f"\nTraining set size: {len(X_train):,}")
        print(f"Testing set size: {len(X_test):,}")
        print(f"Number of features: {len(feature_names)}")
        
        # Get all model configurations
        model_configs = self.get_model_configs()
        
        # Create output directories and clean plots directory
        plots_dir = os.path.join(output_dir, 'Plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Train and evaluate all models
        all_results = {}
        for model_name, (base_model, param_grid) in model_configs.items():
            print("\n" + "="*70)
            print(f"TRAINING {model_name.upper()} WITH CROSS-VALIDATION")
            print("="*70)
            
            # Perform GridSearchCV for hyperparameter tuning
            if param_grid:
                print(f"\nPerforming GridSearchCV with {len(param_grid)} hyperparameters...")
                print(f"Parameter grid: {param_grid}")
                
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='precision',
                    n_jobs=self.n_jobs,
                    verbose=1
                )
                
                start_time = time.time()
                grid_search.fit(X_train, y_train)
                duration = time.time() - start_time
                
                print(f"\nGridSearchCV completed in {duration:.2f} seconds")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
                
                # Use best estimator
                model = grid_search.best_estimator_
                self.best_params[model_name] = grid_search.best_params_
            else:
                # Train without hyperparameter tuning
                start_time = time.time()
                base_model.fit(X_train, y_train)
                duration = time.time() - start_time
                print(f"{model_name} training completed in {duration:.2f} seconds")
                model = base_model
            
            self.models[model_name] = model
            
            # Evaluate model
            results = self.evaluate_model(model, X_test, y_test, model_name)
            all_results[model_name] = results
            
            # Plot confusion matrix for each model
            self.plot_confusion_matrix(
                results['confusion_matrix'], model_name,
                output_path=os.path.join(plots_dir, f'{model_name.lower()}_confusion_matrix.png')
            )
            
            # Plot feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(
                    model, feature_names, model_name,
                    output_path=os.path.join(plots_dir, f'{model_name.lower()}_feature_importance.png')
                )
            
            # Plot decision tree for DecisionTree model
            if model_name == "DecisionTree":
                self.plot_decision_tree(
                    model, feature_names, model_name,
                    output_path=os.path.join(plots_dir, f'{model_name.lower()}_tree.png'),
                    max_depth=3  # Show only top 3 levels for readability
                )
        
        # Save models
        self.save_models(output_dir)
        
        # Save evaluation metrics to CSV
        self.save_evaluation_metrics(output_dir)
        
        print("\n" + "="*70)
        print("MODEL TRAINING PIPELINE COMPLETED")
        print("="*70)
        
        return {
            'models': self.models,
            'all_results': all_results,
            'X_test': X_test,
            'y_test': y_test
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    
    print("="*70)
    print("STANDALONE MODEL TRAINING")
    print("="*70)
    
    # Load pre-split datasets from feature engineering
    data_dir = 'SampleData'
    output_dir = 'Models'
    
    print(f"\nLoading pre-split datasets from: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv.gz'), compression='gzip')
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv.gz'), compression='gzip')
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv.gz'), compression='gzip')['y_train']
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv.gz'), compression='gzip')['y_test']
    
    # Load feature names
    with open(os.path.join(data_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"Loaded datasets:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Number of features: {len(feature_names)}")
    
    # Initialize and run model trainer
    trainer = ModelTrainer(random_state=42, n_jobs=-1)
    
    results = trainer.run_training_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"Models saved to: {output_dir}/")
    print(f"Plots saved to: {output_dir}/Plots/")
