"""
Naive Baseline Model for ASL Recognition
Predicts most frequent class or random
"""
import numpy as np
from collections import Counter
import joblib


class NaiveBaseline:
    """
    Naive baseline that predicts the most frequent class
    """
    def __init__(self, strategy='most_frequent'):
        """
        Args:
            strategy: 'most_frequent' or 'random'
        """
        self.strategy = strategy
        self.most_frequent_class = None
        self.classes = None
    
    def fit(self, y_train):
        """
        Fit the baseline model
        
        Args:
            y_train: training labels
        """
        counter = Counter(y_train)
        self.most_frequent_class = counter.most_common(1)[0][0]
        self.classes = list(set(y_train))
        return self
    
    def predict(self, X):
        """
        Predict classes
        
        Args:
            X: input data (ignored for baseline)
        
        Returns:
            predictions
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        
        if self.strategy == 'most_frequent':
            return np.array([self.most_frequent_class] * n_samples)
        elif self.strategy == 'random':
            return np.random.choice(self.classes, size=n_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def save(self, filepath):
        """Save model"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load model"""
        return joblib.load(filepath)