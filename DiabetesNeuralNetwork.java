/**
 * Artificial Neural Network Implementation in Java
 * For diabetes prediction using CDC health indicators dataset
 */

 import java.io.*;
 import java.util.*;
 import java.nio.file.*;
 import java.util.concurrent.ThreadLocalRandom;
 
 // Define DataPoint class outside the main class
 class DataPoint {
	 double[] features;
	 double target;
	 
	 public DataPoint(double[] features, double target) {
		 this.features = features;
		 this.target = target;
	 }
 }
 
 // Interface for activation functions
 interface ActivationFunction {
	 double activate(double x);
	 double derivative(double x);
 }
 
 // Sigmoid activation function
 class Sigmoid implements ActivationFunction {
	 @Override
	 public double activate(double x) {
		 return 1.0 / (1.0 + Math.exp(-x));
	 }
	 
	 @Override
	 public double derivative(double x) {
		 double sigmoid = activate(x);
		 return sigmoid * (1 - sigmoid);
	 }
 }
 
 // ReLU activation function
 class ReLU implements ActivationFunction {
	 @Override
	 public double activate(double x) {
		 return Math.max(0, x);
	 }
	 
	 @Override
	 public double derivative(double x) {
		 return x > 0 ? 1 : 0;
	 }
 }
 
 // Matrix class for neural network operations
 class Matrix {
	 double[][] data;
	 int rows, cols;
	 
	 // Constructor with rows and cols
	 public Matrix(int rows, int cols) {
		 this.rows = rows;
		 this.cols = cols;
		 this.data = new double[rows][cols];
	 }
	 
	 // Constructor with existing 2D array
	 public Matrix(double[][] data) {
		 this.rows = data.length;
		 this.cols = data[0].length;
		 this.data = data;
	 }
	 
	 // Constructor from 1D array (convert to column matrix)
	 public Matrix(double[] data) {
		 this.rows = data.length;
		 this.cols = 1;
		 this.data = new double[rows][cols];
		 for (int i = 0; i < rows; i++) {
			 this.data[i][0] = data[i];
		 }
	 }
	 
	 // Initialize with random values
	 public void randomize() {
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 // Initialize with small random values
				 this.data[i][j] = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
			 }
		 }
	 }
	 
	 // Matrix multiplication
	 public Matrix multiply(Matrix b) {
		 if (this.cols != b.rows) {
			 throw new IllegalArgumentException("Columns of A must match rows of B");
		 }
		 
		 Matrix result = new Matrix(this.rows, b.cols);
		 
		 for (int i = 0; i < result.rows; i++) {
			 for (int j = 0; j < result.cols; j++) {
				 double sum = 0;
				 for (int k = 0; k < this.cols; k++) {
					 sum += this.data[i][k] * b.data[k][j];
				 }
				 result.data[i][j] = sum;
			 }
		 }
		 
		 return result;
	 }
	 
	 // Element-wise multiplication (Hadamard product)
	 public Matrix hadamard(Matrix b) {
		 if (this.rows != b.rows || this.cols != b.cols) {
			 throw new IllegalArgumentException("Matrices must have same dimensions");
		 }
		 
		 Matrix result = new Matrix(rows, cols);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[i][j] = this.data[i][j] * b.data[i][j];
			 }
		 }
		 
		 return result;
	 }
	 
	 // Add a matrix
	 public Matrix add(Matrix b) {
		 if (this.rows != b.rows || this.cols != b.cols) {
			 throw new IllegalArgumentException("Matrices must have same dimensions");
		 }
		 
		 Matrix result = new Matrix(rows, cols);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[i][j] = this.data[i][j] + b.data[i][j];
			 }
		 }
		 
		 return result;
	 }
	 
	 // Subtract a matrix
	 public Matrix subtract(Matrix b) {
		 if (this.rows != b.rows || this.cols != b.cols) {
			 throw new IllegalArgumentException("Matrices must have same dimensions");
		 }
		 
		 Matrix result = new Matrix(rows, cols);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[i][j] = this.data[i][j] - b.data[i][j];
			 }
		 }
		 
		 return result;
	 }
	 
	 // Transpose matrix
	 public Matrix transpose() {
		 Matrix result = new Matrix(cols, rows);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[j][i] = this.data[i][j];
			 }
		 }
		 
		 return result;
	 }
	 
	 // Apply a function to each element
	 public Matrix map(ActivationFunction func) {
		 Matrix result = new Matrix(rows, cols);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[i][j] = func.activate(this.data[i][j]);
			 }
		 }
		 
		 return result;
	 }
	 
	 // Apply derivative function to each element
	 public Matrix mapDerivative(ActivationFunction func) {
		 Matrix result = new Matrix(rows, cols);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[i][j] = func.derivative(this.data[i][j]);
			 }
		 }
		 
		 return result;
	 }
	 
	 // Scale matrix by a scalar
	 public Matrix scale(double scalar) {
		 Matrix result = new Matrix(rows, cols);
		 
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 result.data[i][j] = this.data[i][j] * scalar;
			 }
		 }
		 
		 return result;
	 }
	 
	 // Convert matrix to array (assuming single column)
	 public double[] toArray() {
		 if (cols != 1) {
			 throw new IllegalArgumentException("Matrix must be a column vector");
		 }
		 
		 double[] result = new double[rows];
		 for (int i = 0; i < rows; i++) {
			 result[i] = data[i][0];
		 }
		 
		 return result;
	 }
 }
 
 // Neural Network class
 class NeuralNetwork {
	 private int inputSize;
	 private int outputSize;
	 private int[] hiddenSizes;
	 private Matrix[] weights;
	 private Matrix[] biases;
	 private Matrix[] layerOutputs;
	 private Matrix[] layerInputs;
	 private ActivationFunction hiddenActivation;
	 private ActivationFunction outputActivation;
	 
	 // Constructor for variable hidden layers
	 public NeuralNetwork(int inputSize, int... layerSizes) {
		 this.inputSize = inputSize;
		 this.outputSize = layerSizes[layerSizes.length - 1];
		 this.hiddenSizes = new int[layerSizes.length - 1];
		 System.arraycopy(layerSizes, 0, hiddenSizes, 0, layerSizes.length - 1);
		 
		 // Initialize activations
		 this.hiddenActivation = new ReLU();
		 this.outputActivation = new Sigmoid();
		 
		 // Initialize weights and biases
		 int layerCount = layerSizes.length;
		 weights = new Matrix[layerCount];
		 biases = new Matrix[layerCount];
		 layerOutputs = new Matrix[layerCount];
		 layerInputs = new Matrix[layerCount];
		 
		 // Input to first hidden layer
		 weights[0] = new Matrix(hiddenSizes[0], inputSize);
		 weights[0].randomize();
		 biases[0] = new Matrix(hiddenSizes[0], 1);
		 biases[0].randomize();
		 
		 // Hidden layers
		 for (int i = 1; i < layerCount - 1; i++) {
			 weights[i] = new Matrix(hiddenSizes[i], hiddenSizes[i-1]);
			 weights[i].randomize();
			 biases[i] = new Matrix(hiddenSizes[i], 1);
			 biases[i].randomize();
		 }
		 
		 // Last hidden layer to output
		 weights[layerCount-1] = new Matrix(outputSize, hiddenSizes[hiddenSizes.length-1]);
		 weights[layerCount-1].randomize();
		 biases[layerCount-1] = new Matrix(outputSize, 1);
		 biases[layerCount-1].randomize();
	 }
	 
	 // Feedforward
	 public double[] predict(double[] inputArray) {
		 // Convert input to matrix
		 Matrix inputs = new Matrix(inputArray);
		 
		 // Input to first hidden layer
		 layerInputs[0] = weights[0].multiply(inputs).add(biases[0]);
		 layerOutputs[0] = layerInputs[0].map(hiddenActivation);
		 
		 // Through hidden layers
		 for (int i = 1; i < weights.length - 1; i++) {
			 layerInputs[i] = weights[i].multiply(layerOutputs[i-1]).add(biases[i]);
			 layerOutputs[i] = layerInputs[i].map(hiddenActivation);
		 }
		 
		 // Final output
		 int lastLayer = weights.length - 1;
		 layerInputs[lastLayer] = weights[lastLayer].multiply(layerOutputs[lastLayer-1]).add(biases[lastLayer]);
		 layerOutputs[lastLayer] = layerInputs[lastLayer].map(outputActivation);
		 
		 // Convert output matrix to array
		 return layerOutputs[lastLayer].toArray();
	 }
	 
	 // Train the network
	 public void train(double[] inputArray, double[] targetArray, double learningRate) {
		 // Forward pass
		 double[] outputArray = predict(inputArray);
		 
		 // Convert to matrices
		 Matrix targets = new Matrix(targetArray);
		 Matrix outputs = new Matrix(outputArray);
		 
		 // Calculate output error
		 Matrix outputErrors = targets.subtract(outputs);
		 
		 // Backpropagation
		 int lastLayer = weights.length - 1;
		 
		 // Output layer gradients
		 Matrix outputGradients = layerInputs[lastLayer].mapDerivative(outputActivation);
		 outputGradients = outputGradients.hadamard(outputErrors).scale(learningRate);
		 
		 // Update output weights and biases
		 Matrix lastHiddenOutputsT = layerOutputs[lastLayer-1].transpose();
		 Matrix outputWeightDeltas = outputGradients.multiply(lastHiddenOutputsT);
		 weights[lastLayer] = weights[lastLayer].add(outputWeightDeltas);
		 biases[lastLayer] = biases[lastLayer].add(outputGradients);
		 
		 // Hidden layers
		 Matrix currentErrors = outputErrors;
		 
		 for (int i = lastLayer - 1; i >= 0; i--) {
			 // Calculate hidden layer errors
			 Matrix weightsT = weights[i+1].transpose();
			 Matrix hiddenErrors = weightsT.multiply(currentErrors);
			 
			 // Calculate gradients
			 Matrix hiddenGradients = layerInputs[i].mapDerivative(hiddenActivation);
			 hiddenGradients = hiddenGradients.hadamard(hiddenErrors).scale(learningRate);
			 
			 // Calculate deltas
			 Matrix prevOutputsT;
			 if (i == 0) {
				 prevOutputsT = new Matrix(inputArray).transpose();
			 } else {
				 prevOutputsT = layerOutputs[i-1].transpose();
			 }
			 
			 Matrix weightDeltas = hiddenGradients.multiply(prevOutputsT);
			 
			 // Update weights and biases
			 weights[i] = weights[i].add(weightDeltas);
			 biases[i] = biases[i].add(hiddenGradients);
			 
			 // Propagate errors backward
			 currentErrors = hiddenErrors;
		 }
	 }
 }
 
 // Main class
 public class DiabetesNeuralNetwork {
	 public static void main(String[] args) throws IOException {
		 // Load and preprocess the data
		 String filePath = "diabetes_012_health_indicators_BRFSS2015.csv";
		 List<DataPoint> dataPoints = loadData(filePath);
		 
		 // Split data into training and testing sets (70/30)
		 Collections.shuffle(dataPoints); // Randomize data order
		 int splitIndex = (int) (dataPoints.size() * 0.7);
		 List<DataPoint> trainingData = dataPoints.subList(0, splitIndex);
		 List<DataPoint> testingData = dataPoints.subList(splitIndex, dataPoints.size());
		 
		 System.out.println("Training data size: " + trainingData.size());
		 System.out.println("Testing data size: " + testingData.size());
		 
		 // Extract training features and labels
		 double[][] trainingFeatures = extractFeatures(trainingData);
		 double[][] trainingLabels = extractLabels(trainingData);
		 
		 // Extract testing features and labels
		 double[][] testingFeatures = extractFeatures(testingData);
		 double[][] testingLabels = extractLabels(testingData);
		 
		 System.out.println("Starting neural network training...");
 
		 // Experiment 1: Vary hidden layer neurons with 2 hidden layers
		 System.out.println("\n--- Experiment 1: Varying hidden layer sizes (2 hidden layers) ---");
		 int[] neuronCounts = {5, 50, 100};
		 for (int neurons : neuronCounts) {
			 System.out.println("\nTraining with " + neurons + " neurons per hidden layer:");
			 NeuralNetwork network = new NeuralNetwork(21, neurons, neurons, 1);
			 trainAndEvaluate(network, trainingFeatures, trainingLabels, testingFeatures, testingLabels);
		 }
		 
		 // Experiment 2: Vary hidden layer count with best neuron count
		 System.out.println("\n--- Experiment 2: Varying hidden layer count ---");
		 testDifferentLayerCounts(trainingFeatures, trainingLabels, testingFeatures, testingLabels);
		 
		 // Experiment 3: Vary learning rate with best architecture
		 System.out.println("\n--- Experiment 3: Varying learning rate ---");
		 testDifferentLearningRates(trainingFeatures, trainingLabels, testingFeatures, testingLabels);
	 }
	 
	 // Load and preprocess the data
	 private static List<DataPoint> loadData(String filePath) throws IOException {
		 List<DataPoint> dataPoints = new ArrayList<>();
		 List<String> lines = Files.readAllLines(Paths.get(filePath));
		 
		 String[] headers = lines.get(0).split(",");
		 
		 // Min-max values for normalization
		 double[] minValues = new double[headers.length-1];
		 double[] maxValues = new double[headers.length-1];
		 Arrays.fill(minValues, Double.MAX_VALUE);
		 Arrays.fill(maxValues, Double.MIN_VALUE);
		 
		 // First pass: find min/max values
		 for (int i = 1; i < lines.size(); i++) {
			 String[] values = lines.get(i).split(",");
			 for (int j = 1; j < values.length; j++) { // Skip the target
				 double val = Double.parseDouble(values[j]);
				 minValues[j-1] = Math.min(minValues[j-1], val);
				 maxValues[j-1] = Math.max(maxValues[j-1], val);
			 }
		 }
		 
		 // Second pass: normalize and create data points
		 for (int i = 1; i < lines.size(); i++) {
			 String[] values = lines.get(i).split(",");
			 
			 // Get the target (diabetes) and convert to binary
			 double diabetesValue = Double.parseDouble(values[0]);
			 double target = diabetesValue > 0 ? 1.0 : 0.0; // Convert to binary target
			 
			 // Get normalized features
			 double[] features = new double[values.length - 1];
			 for (int j = 1; j < values.length; j++) {
				 double val = Double.parseDouble(values[j]);
				 // Normalize to [0,1]
				 if (maxValues[j-1] > minValues[j-1]) {
					 features[j-1] = (val - minValues[j-1]) / (maxValues[j-1] - minValues[j-1]);
				 } else {
					 features[j-1] = 0.5; // Default if min=max
				 }
			 }
			 
			 dataPoints.add(new DataPoint(features, target));
		 }
		 
		 System.out.println("Loaded " + dataPoints.size() + " data points");
		 return dataPoints;
	 }
	 
	 // Extract features from data points
	 private static double[][] extractFeatures(List<DataPoint> dataPoints) {
		 double[][] features = new double[dataPoints.size()][];
		 for (int i = 0; i < dataPoints.size(); i++) {
			 features[i] = dataPoints.get(i).features;
		 }
		 return features;
	 }
	 
	 // Extract labels from data points
	 private static double[][] extractLabels(List<DataPoint> dataPoints) {
		 double[][] labels = new double[dataPoints.size()][1];
		 for (int i = 0; i < dataPoints.size(); i++) {
			 labels[i][0] = dataPoints.get(i).target;
		 }
		 return labels;
	 }
	 
	 // Test with different hidden layer counts
	 private static void testDifferentLayerCounts(double[][] trainingFeatures, double[][] trainingLabels,
												double[][] testingFeatures, double[][] testingLabels) {
		 // Test with 2, 4 and 6 hidden layers with 50 neurons each
		 
		 // 2 hidden layers (already tested above)
		 System.out.println("\nTraining with 2 hidden layers (50 neurons each):");
		 NeuralNetwork network2 = new NeuralNetwork(21, 50, 50, 1);
		 trainAndEvaluate(network2, trainingFeatures, trainingLabels, testingFeatures, testingLabels);
		 
		 // 4 hidden layers
		 System.out.println("\nTraining with 4 hidden layers (50 neurons each):");
		 NeuralNetwork network4 = new NeuralNetwork(21, 50, 50, 50, 50, 1);
		 trainAndEvaluate(network4, trainingFeatures, trainingLabels, testingFeatures, testingLabels);
		 
		 // 6 hidden layers
		 System.out.println("\nTraining with 6 hidden layers (50 neurons each):");
		 NeuralNetwork network6 = new NeuralNetwork(21, 50, 50, 50, 50, 50, 50, 1);
		 trainAndEvaluate(network6, trainingFeatures, trainingLabels, testingFeatures, testingLabels);
	 }
	 
	 // Test with different learning rates
	 private static void testDifferentLearningRates(double[][] trainingFeatures, double[][] trainingLabels,
												 double[][] testingFeatures, double[][] testingLabels) {
		 // Using 4 hidden layers with 50 neurons each (best architecture)
		 double[] learningRates = {0.01, 0.1, 0.5};
		 
		 for (double learningRate : learningRates) {
			 System.out.println("\nTraining with learning rate: " + learningRate);
			 NeuralNetwork network = new NeuralNetwork(21, 50, 50, 50, 50, 1);
			 
			 // Train and evaluate
			 long startTime = System.currentTimeMillis();
			 for (int epoch = 0; epoch < 1000; epoch++) {
				 for (int i = 0; i < trainingFeatures.length; i++) {
					 network.train(trainingFeatures[i], trainingLabels[i], learningRate);
				 }
				 
				 if (epoch % 100 == 0) {
					 double accuracy = evaluateAccuracy(network, trainingFeatures, trainingLabels);
					 System.out.println("Epoch " + epoch + ": Training accuracy = " + accuracy);
				 }
			 }
			 
			 long endTime = System.currentTimeMillis();
			 System.out.println("Training took " + ((endTime - startTime) / 1000.0) + " seconds");
			 
			 // Evaluate on test set
			 double testAccuracy = evaluateAccuracy(network, testingFeatures, testingLabels);
			 System.out.println("Test accuracy: " + testAccuracy);
		 }
	 }
	 
	 // Train and evaluate a neural network
	 private static void trainAndEvaluate(NeuralNetwork network, double[][] trainingFeatures, double[][] trainingLabels,
									   double[][] testingFeatures, double[][] testingLabels) {
		 long startTime = System.currentTimeMillis();
		 
		 // Train for 1000 epochs
		 for (int epoch = 0; epoch < 1000; epoch++) {
			 for (int i = 0; i < trainingFeatures.length; i++) {
				 network.train(trainingFeatures[i], trainingLabels[i], 0.01);
			 }
			 
			 if (epoch % 100 == 0) {
				 double accuracy = evaluateAccuracy(network, trainingFeatures, trainingLabels);
				 System.out.println("Epoch " + epoch + ": Training accuracy = " + accuracy);
			 }
		 }
		 
		 long endTime = System.currentTimeMillis();
		 System.out.println("Training took " + ((endTime - startTime) / 1000.0) + " seconds");
		 
		 // Evaluate on test set
		 double testAccuracy = evaluateAccuracy(network, testingFeatures, testingLabels);
		 System.out.println("Test accuracy: " + testAccuracy);
	 }
	 
	 // Evaluate the accuracy of a neural network
	 private static double evaluateAccuracy(NeuralNetwork network, double[][] features, double[][] labels) {
		 int correct = 0;
		 for (int i = 0; i < features.length; i++) {
			 double[] output = network.predict(features[i]);
			 double prediction = output[0] > 0.5 ? 1.0 : 0.0;
			 if (Math.abs(prediction - labels[i][0]) < 0.01) {
				 correct++;
			 }
		 }
		 return (double) correct / features.length;
	 }
 }

