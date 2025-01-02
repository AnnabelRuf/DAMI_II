import numpy as np
import pandas as pd

# Simulate dataset
def simulate_data(num_samples=100, num_features=401, output_file="simulated_data.csv"):
    np.random.seed(42)  # For reproducibility

    # Generate random feature values
    data = np.random.rand(num_samples, num_features)

    # Assign column names
    feature_columns = [f"feature_{i}" for i in range(1, num_features + 1)]

    # Generate a binary target variable (0 or 1)
    target = np.random.choice([0, 1], size=num_samples)

    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_columns)
    df['target'] = target

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Simulated dataset saved to {output_file}")

# Main script
if __name__ == "__main__":
    simulate_data()
