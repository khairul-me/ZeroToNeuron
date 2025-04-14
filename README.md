# Neural Network for Diabetes Prediction

## Project Overview

This project implements an Artificial Neural Network (ANN) from scratch in Java to predict diabetes using the CDC health indicators dataset. The neural network uses a backpropagation algorithm and supports variable architectures with different hidden layer configurations and learning rates.

## Dataset

The CDC diabetes health indicators dataset contains 253,680 records with health-related features. The target variable `Diabetes_012` is converted to binary for classification (0: No diabetes, 1: Diabetes/Prediabetes).

Dataset source: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The dataset has a significant class imbalance:
- Class 0 (No diabetes): 84.2% of samples
- Class 1 (Diabetes/Prediabetes): 15.8% of samples

## Features

- Complete neural network implementation in Java
- Flexible architecture with variable hidden layers and neurons
- Matrix operations for efficient computations
- ReLU activation for hidden layers and Sigmoid for output layer
- Configurable learning rates
- Experimental framework for testing different configurations

## Project Structure

```
.
├── DiabetesNeuralNetwork.java      # Main class with experiments
├── DataPoint.java                  # Class for storing feature vectors and targets
├── Matrix.java                     # Matrix operations implementation
├── ActivationFunction.java         # Interface for activation functions
├── Sigmoid.java                    # Sigmoid activation implementation
├── ReLU.java                       # ReLU activation implementation
├── NeuralNetwork.java              # Neural network implementation
└── diabetes_012_health_indicators_BRFSS2015.csv  # Dataset
```

## Requirements

- Java Development Kit (JDK) 8 or higher
- CSV file with the diabetes dataset

## Setup and Installation

1. Clone the repository or download the source files
2. Place the diabetes dataset CSV file in the project directory
3. Compile the Java files:

```bash
javac *.java
```

4. Run the main class:

```bash
java DiabetesNeuralNetwork
```

## Usage

The `DiabetesNeuralNetwork` class contains the main method that:
1. Loads and preprocesses the dataset
2. Splits data into training and testing sets
3. Runs three experiments with different neural network configurations

### Creating a Neural Network

```java
// Create a neural network with 21 input features, 
// two hidden layers with 50 neurons each, and 1 output neuron
NeuralNetwork network = new NeuralNetwork(21, 50, 50, 1);
```

### Training the Network

```java
// Train the network with features, labels, and learning rate
network.train(features, labels, 0.01);
```

### Making Predictions

```java
// Get prediction
double[] output = network.predict(features);
```

## Experiments

The project includes three experiments:

### 1. Varying Hidden Layer Size (2 Hidden Layers)

This experiment tests different numbers of neurons (5, 50, 100) in each hidden layer.

| Hidden Layer Size | Training Accuracy | Testing Accuracy | Training Time (s) |
|-------------------|-------------------|------------------|-------------------|
| 5-5               | 84.20%            | 84.33%           | 1,863.6           |
| 50-50             | 84.20%            | 84.33%           | 12,797.8          |
| 100-100           | 84.20%            | 84.33%           | 29,436.8          |

### 2. Varying Hidden Layer Count

This experiment tests different numbers of hidden layers (2, 4, 6) with 50 neurons each.

| Hidden Layers | Configuration       | Testing Accuracy | Training Time (s) |
|---------------|---------------------|------------------|-------------------|
| 2             | 50-50               | 84.33%           | 12,797.8          |
| 4             | 50-50-50-50         | 84.33%           | 21,532.0          |
| 6             | 50-50-50-50-50-50   | 84.33%           | 47,623.8          |

### 3. Varying Learning Rate

This experiment tests different learning rates (0.01, 0.1, 0.5) with 4 hidden layers.

| Learning Rate | Testing Accuracy | Training Time (s) |
|---------------|------------------|-------------------|
| 0.01          | 84.33%           | 17,173.1          |
| 0.1           | 84.33%           | 21,877.9          |
| 0.5           | 84.33%           | 32,187.4          |

## Performance Analysis

The class imbalance in the dataset (84.2% no diabetes vs. 15.8% diabetes/prediabetes) significantly affects the learning process. The model achieves consistent accuracy of ~84.3% across all configurations, suggesting it defaults to predicting the majority class.

Training times vary significantly based on network complexity:
- 5 neurons per layer: ~31 minutes
- 100 neurons per layer: ~8 hours
- 6 hidden layers: ~13 hours

## Neural Network Architecture

The network architecture implemented in this project consists of:
- Input layer: 21 neurons (one for each feature)
- Hidden layers: Variable number (2, 4, or 6 in experiments)
- Hidden layer neurons: Variable count (5, 50, or 100 in experiments) 
- Output layer: 1 neuron with sigmoid activation
- Hidden layer activation: ReLU
- Weight initialization: Random values between -0.5 and 0.5

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**: Increase Java heap space with `-Xmx` flag
   ```
   java -Xmx4g DiabetesNeuralNetwork
   ```

2. **File Not Found**: Ensure the CSV file is in the correct location and named properly

3. **Long Training Time**: Use smaller subsets of data for quick testing by modifying:
   ```java
   // Reduce to smaller sample size for testing
   int sampleSize = 10000;
   List<DataPoint> sampledData = dataPoints.subList(0, Math.min(sampleSize, dataPoints.size()));
   ```

## Future Improvements

1. Address class imbalance through resampling or cost-sensitive learning
2. Implement regularization (L1/L2) to prevent overfitting
3. Add batch normalization for more stable training
4. Implement mini-batch gradient descent
5. Add early stopping based on validation performance

## References

1. Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.
2. "CDC Diabetes Health Indicators Dataset," UCI Machine Learning Repository.
3. "Neural Network Tutorial - Developing a Neural Network from Scratch," YouTube, Coding With John. [Tutorial Link](https://www.youtube.com/watch?v=3MMonOWGe0M&list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN)

## License

This project is provided for educational purposes only. The CDC dataset is publicly available under its own terms of use.
