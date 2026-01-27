"""ML models for text classification"""

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
    PRIORITY_MODEL_PATH, PRIORITY_VECTORIZER_PATH,
    ISSUE_TYPE_MODEL_PATH, ISSUE_TYPE_VECTORIZER_PATH,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER, TEST_SIZE, RANDOM_STATE
)


class TextClassificationModel:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None
    
    def train(self, X: np.ndarray, y: np.ndarray, evaluate: bool = True) -> dict:
        metrics = {}
        
        if evaluate:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words='english'
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        self.model = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_MAX_ITER,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        self.model.fit(X_train_tfidf, y_train)
        self.classes_ = self.model.classes_
        
        train_pred = self.model.predict(X_train_tfidf)
        metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        metrics['train_samples'] = len(y_train)
        
        if evaluate and X_test is not None:
            X_test_tfidf = self.vectorizer.transform(X_test)
            test_pred = self.model.predict(X_test_tfidf)
            metrics['test_accuracy'] = accuracy_score(y_test, test_pred)
            metrics['test_samples'] = len(y_test)
            metrics['classification_report'] = classification_report(y_test, test_pred, output_dict=True)
        
        return metrics
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded")
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)
    
    def predict_single(self, text: str) -> str:
        return self.predict([text])[0]
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded")
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)
    
    def get_prediction_confidence(self, text: str) -> Tuple[str, float]:
        probs = self.predict_proba([text])[0]
        idx = np.argmax(probs)
        return self.classes_[idx], probs[idx]
    
    def save(self) -> None:
        if self.model is None or self.vectorizer is None:
            raise ValueError("No model to save")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Saved to {self.model_path}")
    
    def load(self) -> bool:
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.classes_ = self.model.classes_
            return True
        except FileNotFoundError:
            return False


class PriorityModel(TextClassificationModel):
    def __init__(self):
        super().__init__(PRIORITY_MODEL_PATH, PRIORITY_VECTORIZER_PATH)


class IssueTypeModel(TextClassificationModel):
    def __init__(self):
        super().__init__(ISSUE_TYPE_MODEL_PATH, ISSUE_TYPE_VECTORIZER_PATH)


def get_top_features_for_class(vectorizer: TfidfVectorizer, model: LogisticRegression, 
                                class_name: str, n_features: int = 10) -> List[Tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    class_idx = list(model.classes_).index(class_name)
    
    if len(model.classes_) == 2:
        coefficients = model.coef_[0]
        if class_idx == 0:
            coefficients = -coefficients
    else:
        coefficients = model.coef_[class_idx]
    
    top_indices = np.argsort(coefficients)[-n_features:][::-1]
    return [(feature_names[i], coefficients[i]) for i in top_indices]


if __name__ == "__main__":
    print("Models module loaded")
