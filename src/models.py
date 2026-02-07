# - Model training and evaluation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from .feature_engineering import FeatureExtractor, prepare_feature_matrix


class SpamClassifier:
    """
    Spam classifier with multiple algorithm support
    """
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize classifier
        
        Args:
            model_type (str): Type of classifier ('random_forest', 'svm', 'logistic', 'naive_bayes', 'gradient_boosting')
            random_state (int): Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.vectorizers = None
        self.feature_importances_ = None
        
        # Define model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=random_state,
                class_weight='balanced'
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced',
                solver='liblinear'
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y, vectorizers=None):
        """
        Train the classifier
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Labels
            vectorizers (dict, optional): Vectorizers for feature extraction
        """
        self.vectorizers = vectorizers
        self.model.fit(X, y)
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_[0])
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            array: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            array: Prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for score in scoring:
            if score == 'roc_auc' and self.model_type == 'svm':
                # SVM needs probability=True for ROC AUC
                continue
            try:
                scores = cross_val_score(self.model, X, y, cv=cv, scoring=score, n_jobs=-1)
                cv_results[score] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores.tolist()
                }
            except:
                pass
        
        return cv_results
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=3):
        """
        Perform hyperparameter tuning
        
        Args:
            X (array-like): Features
            y (array-like): Labels
            param_grid (dict, optional): Parameter grid
            cv (int): Number of folds
            
        Returns:
            GridSearchCV: Tuned model
        """
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            elif self.model_type == 'logistic':
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                }
            elif self.model_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        
        return grid_search
    
    def plot_feature_importances(self, top_n=20, figsize=(12, 8)):
        """
        Plot feature importances
        
        Args:
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        if self.feature_importances_ is None:
            print("No feature importances available")
            return
        
        # Get feature names
        feature_names = []
        if self.vectorizers and 'text_features' in self.vectorizers:
            feature_names.extend(self.vectorizers['text_features'])
        if self.vectorizers and 'ngram_vectorizer' in self.vectorizers:
            feature_names.extend(self.vectorizers['ngram_vectorizer'].get_feature_names_out())
        
        if len(feature_names) != len(self.feature_importances_):
            print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match "
                  f"number of importances ({len(self.feature_importances_)})")
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importances - {self.model_type}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def save_model(self, model_path, vectorizer_path=None):
        """
        Save model and vectorizers
        
        Args:
            model_path (str): Path to save model
            vectorizer_path (str, optional): Path to save vectorizers
        """
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save vectorizers if provided
        if vectorizer_path and self.vectorizers:
            joblib.dump(self.vectorizers, vectorizer_path)
            print(f"Vectorizers saved to {vectorizer_path}")
    
    @classmethod
    def load_model(cls, model_path, vectorizer_path=None):
        """
        Load model and vectorizers
        
        Args:
            model_path (str): Path to load model from
            vectorizer_path (str, optional): Path to load vectorizers from
            
        Returns:
            SpamClassifier: Loaded classifier
        """
        # Load model
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load vectorizers if provided
        vectorizers = None
        if vectorizer_path:
            vectorizers = joblib.load(vectorizer_path)
            print(f"Vectorizers loaded from {vectorizer_path}")
        
        # Create classifier instance
        classifier = cls()
        classifier.model = model
        classifier.vectorizers = vectorizers
        
        return classifier


def train_test_evaluate(df, text_column='message', label_column='label', 
                       test_size=0.2, model_type='random_forest',
                       feature_type='all'):
    """
    Complete pipeline: train, test, evaluate
    
    Args:
        df (pd.DataFrame): Input data
        text_column (str): Text column name
        label_column (str): Label column name
        test_size (float): Test set size
        model_type (str): Type of model
        feature_type (str): Type of features
        
    Returns:
        tuple: (classifier, metrics, X_test, y_test)
    """
    # Prepare features
    X, y, vectorizers = prepare_feature_matrix(
        df, text_column, label_column, feature_type
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train classifier
    classifier = SpamClassifier(model_type=model_type)
    classifier.fit(X_train, y_train, vectorizers)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Model: {model_type}")
    print(f"Features: {feature_type}")
    print(f"Test Size: {test_size}")
    print(f"{'='*50}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"{'='*50}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    return classifier, metrics, X_test, y_test


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    data = {
        'message': [
            "WINNER!! You've won a free iPhone! Call now to claim.",
            "Hi John, are we meeting tomorrow for lunch?",
            "URGENT: Your bank account has been compromised.",
            "Don't forget to bring the documents to the meeting.",
            "FREE entry to win £1000 cash prize! Text WIN to 88888.",
            "Hey, what time are we catching up tonight?",
            "Congratulations! You've been selected for a prize.",
            "Meeting rescheduled to 3 PM tomorrow.",
            "Claim your bonus reward now! Limited time offer.",
            "See you at the cinema at 7 PM."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
    }
    
    df = pd.DataFrame(data)
    
    # Train and evaluate
    classifier, metrics, X_test, y_test = train_test_evaluate(
        df, 
        model_type='random_forest',
        feature_type='all'
    )
    
    # Plot feature importances
    classifier.plot_feature_importances(top_n=10)
    
    # Save model
    classifier.save_model('spam_classifier.pkl', 'vectorizers.pkl')