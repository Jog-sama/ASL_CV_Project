"""
Classical ML Model using Hand Landmarks + Random Forest
"""
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_features(self, image):
        """
        Extract hand landmark features from image
        
        Args:
            image: RGB image array
        
        Returns:
            feature vector (63 dimensions: 21 landmarks x 3 coordinates)
        """
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            features = []
            
            for landmark in landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(features)
        else:
            # Return zeros if no hand detected
            return np.zeros(63)
    
    def extract_batch(self, images):
        """Extract features from batch of images"""
        features = []
        for img in tqdm(images, desc="Extracting landmarks"):
            feat = self.extract_features(img)
            features.append(feat)
        return np.array(features)
    
    def close(self):
        """Clean up resources"""
        self.hands.close()


class ClassicalASLModel:
    """
    Classical ML model: Hand Landmarks + Random Forest
    """
    def __init__(self, n_estimators=100):
        self.extractor = HandLandmarkExtractor()
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: training images (RGB)
            y_train: training labels
        """
        print("Extracting training features...")
        features = self.extractor.extract_batch(X_train)
        
        print("Scaling features...")
        features_scaled = self.scaler.fit_transform(features)
        
        print("Training Random Forest...")
        self.classifier.fit(features_scaled, y_train)
        
        return self
    
    def predict(self, X):
        """
        Predict classes
        
        Args:
            X: images (RGB)
        
        Returns:
            predictions
        """
        features = self.extractor.extract_batch(X)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        features = self.extractor.extract_batch(X)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict_proba(features_scaled)
    
    def save(self, filepath):
        """Save model"""
        model_dict = {
            'scaler': self.scaler,
            'classifier': self.classifier
        }
        joblib.dump(model_dict, filepath)
    
    def close(self):
        """Clean up resources"""
        self.extractor.close()
    
    @staticmethod
    def load(filepath):
        """Load model"""
        model_dict = joblib.load(filepath)
        model = ClassicalASLModel()
        model.scaler = model_dict['scaler']
        model.classifier = model_dict['classifier']
        return model