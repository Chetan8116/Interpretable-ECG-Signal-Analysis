import * as tf from '@tensorflow/tfjs';

export interface MLPConfig {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  learningRate: number;
  activation: 'relu' | 'sigmoid' | 'tanh';
}

export interface ECGPrediction {
  condition: string;
  confidence: number;
  probabilities: { [key: string]: number };
  features: number[];
}

export class MLPModel {
  private model: tf.Sequential | null = null;
  private config: MLPConfig;
  private isTraining: boolean = false;
  
  // ECG condition labels
  private readonly labels = [
    'Normal Sinus Rhythm',
    'Atrial Fibrillation',
    'Atrial Flutter',
    'Premature Ventricular Contraction',
    'Ventricular Tachycardia',
    'ST Elevation',
    'ST Depression',
    'Left Bundle Branch Block',
    'Right Bundle Branch Block',
    'Myocardial Infarction'
  ];

  constructor(config?: Partial<MLPConfig>) {
    this.config = {
      inputSize: 500, // 500 samples per ECG segment
      hiddenLayers: [256, 128, 64, 32],
      outputSize: 10, // 10 conditions
      learningRate: 0.001,
      activation: 'relu',
      ...config
    };
  }

  /**
   * Build the MLP neural network architecture
   */
  async buildModel(): Promise<void> {
    this.model = tf.sequential();

    // Input layer
    this.model.add(tf.layers.dense({
      inputShape: [this.config.inputSize],
      units: this.config.hiddenLayers[0],
      activation: this.config.activation,
      kernelInitializer: 'heNormal',
      name: 'input_layer'
    }));

    // Add batch normalization for stability
    this.model.add(tf.layers.batchNormalization());
    this.model.add(tf.layers.dropout({ rate: 0.3 }));

    // Hidden layers with dropout for regularization
    this.config.hiddenLayers.slice(1).forEach((units, index) => {
      this.model!.add(tf.layers.dense({
        units,
        activation: this.config.activation,
        kernelInitializer: 'heNormal',
        name: `hidden_layer_${index + 1}`
      }));
      
      this.model!.add(tf.layers.batchNormalization());
      this.model!.add(tf.layers.dropout({ rate: 0.2 }));
    });

    // Output layer with softmax for multi-class classification
    this.model.add(tf.layers.dense({
      units: this.config.outputSize,
      activation: 'softmax',
      name: 'output_layer'
    }));

    // Compile model with Adam optimizer
    this.model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    console.log('MLP Model built successfully');
    this.model.summary();
  }

  /**
   * Extract features from raw ECG signal
   */
  private extractFeatures(ecgData: number[]): number[] {
    // Normalize the signal
    const mean = ecgData.reduce((a, b) => a + b, 0) / ecgData.length;
    const std = Math.sqrt(
      ecgData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / ecgData.length
    );
    
    const normalized = ecgData.map(val => (val - mean) / (std || 1));

    // Calculate statistical features
    const features = [];
    
    // Time-domain features
    features.push(Math.max(...normalized));
    features.push(Math.min(...normalized));
    features.push(mean);
    features.push(std);
    
    // RR interval features (heart rate variability)
    const peaks = this.detectRPeaks(normalized);
    const rrIntervals = [];
    for (let i = 1; i < peaks.length; i++) {
      rrIntervals.push(peaks[i] - peaks[i - 1]);
    }
    
    if (rrIntervals.length > 0) {
      const rrMean = rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length;
      const rrStd = Math.sqrt(
        rrIntervals.reduce((sum, val) => sum + Math.pow(val - rrMean, 2), 0) / rrIntervals.length
      );
      features.push(rrMean);
      features.push(rrStd);
    } else {
      features.push(0, 0);
    }

    // QRS complex features
    const qrsWidth = this.calculateQRSWidth(normalized);
    features.push(qrsWidth);

    // Pad or truncate to input size
    while (features.length < this.config.inputSize) {
      features.push(0);
    }

    return features.slice(0, this.config.inputSize);
  }

  /**
   * Detect R-peaks in ECG signal (simplified Pan-Tompkins algorithm)
   */
  private detectRPeaks(signal: number[]): number[] {
    const peaks: number[] = [];
    const threshold = Math.max(...signal) * 0.6;
    
    for (let i = 1; i < signal.length - 1; i++) {
      if (signal[i] > threshold && 
          signal[i] > signal[i - 1] && 
          signal[i] > signal[i + 1]) {
        // Ensure minimum distance between peaks (refractory period)
        if (peaks.length === 0 || i - peaks[peaks.length - 1] > 50) {
          peaks.push(i);
        }
      }
    }
    
    return peaks;
  }

  /**
   * Calculate QRS complex width
   */
  private calculateQRSWidth(signal: number[]): number {
    const peaks = this.detectRPeaks(signal);
    if (peaks.length === 0) return 0;
    
    // Simple QRS width estimation
    const qrsWidths = peaks.map(peak => {
      let start = peak;
      let end = peak;
      const threshold = signal[peak] * 0.3;
      
      while (start > 0 && signal[start] > threshold) start--;
      while (end < signal.length - 1 && signal[end] > threshold) end++;
      
      return end - start;
    });
    
    return qrsWidths.reduce((a, b) => a + b, 0) / qrsWidths.length;
  }

  /**
   * Predict ECG condition using the trained model
   */
  async predict(ecgData: number[]): Promise<ECGPrediction> {
    if (!this.model) {
      throw new Error('Model not initialized. Call buildModel() first.');
    }

    // Extract features
    const features = this.extractFeatures(ecgData);
    
    // Make prediction
    const inputTensor = tf.tensor2d([features]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const probabilities = await prediction.data();
    
    // Get predicted class
    const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
    const condition = this.labels[maxProbIndex];
    const confidence = probabilities[maxProbIndex];

    // Create probabilities object
    const probabilitiesObj: { [key: string]: number } = {};
    this.labels.forEach((label, index) => {
      probabilitiesObj[label] = probabilities[index];
    });

    // Cleanup tensors
    inputTensor.dispose();
    prediction.dispose();

    return {
      condition,
      confidence,
      probabilities: probabilitiesObj,
      features: features.slice(0, 10) // Return first 10 features for visualization
    };
  }

  /**
   * Train the model with sample data (for demonstration)
   */
  async trainModel(
    trainingData: number[][],
    labels: number[][],
    epochs: number = 50,
    batchSize: number = 32,
    validationSplit: number = 0.2
  ): Promise<void> {
    if (!this.model) {
      throw new Error('Model not initialized. Call buildModel() first.');
    }

    this.isTraining = true;

    const xTrain = tf.tensor2d(trainingData);
    const yTrain = tf.tensor2d(labels);

    try {
      const history = await this.model.fit(xTrain, yTrain, {
        epochs,
        batchSize,
        validationSplit,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(
              `Epoch ${epoch + 1}/${epochs} - ` +
              `loss: ${logs?.loss.toFixed(4)} - ` +
              `accuracy: ${logs?.acc.toFixed(4)} - ` +
              `val_loss: ${logs?.val_loss.toFixed(4)} - ` +
              `val_accuracy: ${logs?.val_acc.toFixed(4)}`
            );
          }
        }
      });

      console.log('Training completed successfully');
    } finally {
      xTrain.dispose();
      yTrain.dispose();
      this.isTraining = false;
    }
  }

  /**
   * Save model to browser storage
   */
  async saveModel(name: string = 'ecg-mlp-model'): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }
    
    await this.model.save(`localstorage://${name}`);
    console.log('Model saved successfully');
  }

  /**
   * Load model from browser storage
   */
  async loadModel(name: string = 'ecg-mlp-model'): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(`localstorage://${name}`) as tf.Sequential;
      console.log('Model loaded successfully');
    } catch (error) {
      console.warn('Could not load saved model, building new model');
      await this.buildModel();
    }
  }

  /**
   * Get model summary
   */
  getSummary(): string {
    if (!this.model) {
      return 'Model not initialized';
    }
    
    let summary = 'MLP Model Architecture:\n';
    summary += `Input Size: ${this.config.inputSize}\n`;
    summary += `Hidden Layers: ${this.config.hiddenLayers.join(' -> ')}\n`;
    summary += `Output Size: ${this.config.outputSize}\n`;
    summary += `Total Parameters: ${this.model.countParams()}\n`;
    
    return summary;
  }

  /**
   * Dispose of the model and free memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

export default MLPModel;
