import numpy as np
from src.predictor import InfraHealthPredictor

def run_prediction():
    # Simulated historical CPU usage trend over 10 hours (percentage)
    time_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cpu_utilization = np.array([45.0, 48.5, 52.0, 56.1, 61.0, 65.2, 70.0, 74.8, 79.1, 84.0])

    predictor = InfraHealthPredictor()
    predictor.train_model(time_hours, cpu_utilization)
    
    critical_hour = predictor.predict_exhaustion_time(threshold=95.0)

    print("\n=============================================")
    print("   AI INFRASTRUCTURE PREDICTIVE MAINTENANCE  ")
    print("=============================================\n")
    print(f"Current CPU Trend:  Linear Increase (~4.3% / hour)")
    print(f"Critical Threshold: 95.0% Resource Limit")
    print(f"⚠️ Predicted Critical Resource Exhaustion at Hour: {critical_hour}")

if __name__ == "__main__":
    run_prediction()
  
