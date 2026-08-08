import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class InfraHealthPredictor:
    """Predicts system resource exhaustion trends using linear trend forecasting."""
    
    def __init__(self):
        self.model = LinearRegression()

    def train_model(self, time_steps: np.ndarray, cpu_usage: np.ndarray):
        X = time_steps.reshape(-1, 1)
        self.model.fit(X, cpu_usage)

    def predict_exhaustion_time(self, threshold: float = 95.0) -> float:
        """Calculates the estimated time step when CPU/RAM reaches critical limit."""
        slope = self.model.coef_[0]
        intercept = self.model.intercept_
        
        if slope <= 0:
            return float('inf')  # Metric stable or declining
            
        time_to_threshold = (threshold - intercept) / slope
        return round(float(time_to_threshold), 2)
      
