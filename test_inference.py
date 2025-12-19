# test_model_inference.py
"""
Test script to verify both saved models can be loaded and used for inference.
"""

import joblib
import pandas as pd
from pathlib import Path


def test_model(model_path):
    """Load model and test inference."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('='*60)
    
    try:
        # Load model
        print("Loading model...", end=" ")
        model = joblib.load(model_path)
        print("✓")
        
        if hasattr(model, 'named_steps'):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'credit_score': [619, 608],
            'geography': ['France', 'Spain'],
            'gender': ['Female', 'Female'],
            'age': [42, 41],
            'tenure': [2, 1],
            'balance': [0.00, 83807.86],
            'num_of_products': [1, 1],
            'has_cr_card': [1, 0],
            'is_active_member': [1, 1],
            'estimated_salary': [101348.8, 112542.8],
            'customer_id': [14522313, 15544565],
            'surname': ['Boni', 'David']

        })
        
        # Predict
        print("Running inference...", end=" ")
        predictions = model.predict(sample_data)
        print("✓")
        
        # Show results
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  Sample {i}: ${pred:,.2f}")
        
        print(f"\n✓ {Path(model_path).name} - SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def main():
    """Test both models."""
    models = [
        "models/global_best_model.pkl",
        "models/global_best_model_optuna.pkl"
    ]
    
    print("Testing both models...")
    results = [test_model(m) for m in models]
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(results)}/{len(results)} models passed")
    print('='*60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)