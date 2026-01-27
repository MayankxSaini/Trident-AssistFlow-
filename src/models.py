"""
AssistFlow AI - ML Models Module

WORKFLOW STEP 2 & 3: PRIORITY PREDICTION & ISSUE TYPE PREDICTION

Model 1 - Priority Prediction (MANDATORY):
- Input: full_text
- Output: predicted_priority ∈ {Low, Medium, High, Critical}
- Method: TF-IDF + Logistic Regression

Model 2 - Issue Type Prediction (OPTIONAL):
- Input: full_text
- Output: issue_type (Billing, Technical, Account, General, etc.)
- Method: TF-IDF + Logistic Regression

IMPORTANT: These models are simple and explainable.
NO deep learning. NO neural networks.
"""

import pickle
from typing import Optional, Tuple, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import sys
sys.path.append(".")
from config import (
    PRIORITY_MODEL_PATH,
    PRIORITY_VECTORIZER_PATH,
    ISSUE_TYPE_MODEL_PATH,
    ISSUE_TYPE_VECTORIZER_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER,
    TEST_SIZE,
    RANDOM_STATE
)


class TextClassificationModel:
    """
    Simple text classification model using TF-IDF + Logistic Regression.
    
    WHY this approach:
    - TF-IDF captures word importance relative to document corpus
    - Logistic Regression is interpretable and works well for text
    - No black box - we can explain predictions via feature weights
    """
    
    def __init__(self, model_path: str, vectorizer_path: str):
        """
        Initialize the model with paths for saving/loading.
        
        Args:
            model_path: Path to save/load the trained model
            vectorizer_path: Path to save/load the TF-IDF vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None
    
    def train(self, X: np.ndarray, y: np.ndarray, evaluate: bool = True) -> dict:
        """
        Train the model on provided text data.
        
        Args:
            X: Array of text documents
            y: Array of labels
            evaluate: Whether to split data and evaluate performance
            
        Returns:
            Dictionary containing training metrics
        """
        metrics = {}
        
        if evaluate:
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE,
                stratify=y  # Maintain class distribution
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Initialize and fit TF-IDF vectorizer
        # WHY these parameters:
        # - max_features: Limit vocabulary to most important terms
        # - ngram_range: Include bigrams to capture phrases like "not working"
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words='english'  # Remove common words that don't help
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train Logistic Regression
        # WHY Logistic Regression:
        # - Multi-class support via one-vs-rest
        # - Probability outputs for confidence scores
        # - Fast training and prediction
        self.model = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_MAX_ITER,
            random_state=RANDOM_STATE,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.model.fit(X_train_tfidf, y_train)
        self.classes_ = self.model.classes_
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_tfidf)
        metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        metrics['train_samples'] = len(y_train)
        
        # Evaluate on test set if available
        if evaluate and X_test is not None:
            X_test_tfidf = self.vectorizer.transform(X_test)
            test_pred = self.model.predict(X_test_tfidf)
            
            metrics['test_accuracy'] = accuracy_score(y_test, test_pred)
            metrics['test_samples'] = len(y_test)
            metrics['classification_report'] = classification_report(
                y_test, test_pred, output_dict=True
            )
        
        return metrics
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for given texts.
        
        Args:
            texts: List of text documents to classify
            
        Returns:
            Array of predicted labels
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        X_tfidf = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_tfidf)
        
        return predictions
    
    def predict_single(self, text: str) -> str:
        """
        Predict label for a single text document.
        
        Args:
            text: Single text document
            
        Returns:
            Predicted label
        """
        predictions = self.predict([text])
        return predictions[0]
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for given texts.
        
        WHY: Probability scores help explain confidence in predictions.
        
        Args:
            texts: List of text documents
            
        Returns:
            Array of probability distributions over classes
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        X_tfidf = self.vectorizer.transform(texts)
        probabilities = self.model.predict_proba(X_tfidf)
        
        return probabilities
    
    def get_prediction_confidence(self, text: str) -> Tuple[str, float]:
        """
        Get prediction and confidence score for a single text.
        
        Args:
            text: Single text document
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        probabilities = self.predict_proba([text])[0]
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.classes_[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return predicted_label, confidence
    
    def save(self) -> None:
        """
        Save the trained model and vectorizer to disk.
        
        WHY: Persistence allows reuse without retraining.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("No model to save. Train a model first.")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")
    
    def load(self) -> bool:
        """
        Load a previously trained model and vectorizer from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.classes_ = self.model.classes_
            return True
        
        except FileNotFoundError:
            print(f"Model files not found. Please train the model first.")
            return False


# =============================================================================
# CONVENIENCE CLASSES FOR SPECIFIC MODELS
# =============================================================================

class PriorityModel(TextClassificationModel):
    """
    Model 1: Priority Prediction Model
    
    Predicts ticket priority: Low, Medium, High, Critical
    This is the MANDATORY model for AssistFlow AI.
    """
    
    def __init__(self):
        super().__init__(
            model_path=PRIORITY_MODEL_PATH,
            vectorizer_path=PRIORITY_VECTORIZER_PATH
        )


class IssueTypeModel(TextClassificationModel):
    """
    Model 2: Issue Type Prediction Model
    
    Predicts issue type: Billing, Technical, Account, General, etc.
    This is an OPTIONAL model that supports routing and explanation.
    It NEVER overrides priority decisions.
    """
    
    def __init__(self):
        super().__init__(
            model_path=ISSUE_TYPE_MODEL_PATH,
            vectorizer_path=ISSUE_TYPE_VECTORIZER_PATH
        )


def get_top_features_for_class(
    vectorizer: TfidfVectorizer, 
    model: LogisticRegression, 
    class_name: str, 
    n_features: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the most important features (words) for a given class.
    
    WHY: This enables explainability - we can tell users WHY a 
    certain priority was predicted based on which words mattered.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained Logistic Regression model
        class_name: The class to get features for
        n_features: Number of top features to return
        
    Returns:
        List of (feature_name, weight) tuples
    """
    feature_names = vectorizer.get_feature_names_out()
    class_idx = list(model.classes_).index(class_name)
    
    # Get coefficients for this class
    if len(model.classes_) == 2:
        # Binary classification - only one set of coefficients
        coefficients = model.coef_[0]
        if class_idx == 0:
            coefficients = -coefficients
    else:
        # Multi-class - one set per class
        coefficients = model.coef_[class_idx]
    
    # Get top positive features (words that increase probability of this class)
    top_indices = np.argsort(coefficients)[-n_features:][::-1]
    top_features = [(feature_names[i], coefficients[i]) for i in top_indices]
    
    return top_features


if __name__ == "__main__":
    # Quick demonstration
    print("ML Models Module Loaded Successfully")
    print("Available classes: PriorityModel, IssueTypeModel")
