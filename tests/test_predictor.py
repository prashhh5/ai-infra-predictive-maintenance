import numpy as np
from src.predictor import InfraHealthPredictor

def test_exhaustion_prediction():
    predictor = InfraHealthPredictor()
    hours = np.array([1, 2, 3, 4])
    usage = np.array([20.0, 40.0, 60.0, 80.0]) # 20% increase per hour
    
    predictor.train_model(hours, usage)
    prediction = predictor.predict_exhaustion_time(threshold=100.0)
    assert prediction == 5.0

def test_stable_metric_no_exhaustion():
    predictor = InfraHealthPredictor()
    hours = np.array([1, 2, 3])
    usage = np.array([50.0, 48.0, 45.0]) # Decreasing
    
    predictor.train_model(hours, usage)
    assert predictor.predict_exhaustion_time(100.0) == float('inf')
  
