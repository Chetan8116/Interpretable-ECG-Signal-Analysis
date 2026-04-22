import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';
import MLPModel, { ECGPrediction } from '../models/MLPModel';

const MLPAnalyzer: React.FC = () => {
  const { leads, analysisResults, setAnalysisResults, isRecording } = useECGStore();
  const [model] = useState(() => new MLPModel());
  const [isInitialized, setIsInitialized] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    const initModel = async () => {
      try {
        await model.loadModel();
        setIsInitialized(true);
      } catch (error) {
        console.error('Error initializing model:', error);
      }
    };

    initModel();

    return () => {
      model.dispose();
    };
  }, [model]);

  useEffect(() => {
    if (!isRecording || !isInitialized || leads[1].data.length < 500) return;

    const analyzeInterval = setInterval(async () => {
      setIsAnalyzing(true);
      try {
        // Use Lead II data for analysis (most common for rhythm analysis)
        const ecgData = leads[1].data.slice(-500);
        const prediction = await model.predict(ecgData);
        setAnalysisResults(prediction);
      } catch (error) {
        console.error('Analysis error:', error);
      } finally {
        setIsAnalyzing(false);
      }
    }, 3000); // Analyze every 3 seconds

    return () => clearInterval(analyzeInterval);
  }, [isRecording, isInitialized, leads, model, setAnalysisResults]);

  if (!isInitialized) {
    return (
      <div className="medical-panel p-6">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
          <p className="text-gray-400">Initializing MLP Neural Network...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="medical-panel p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">
            MLP Neural Network Analysis
          </h2>
          <p className="text-gray-400 text-sm">
            Multi-Layer Perceptron for ECG Pattern Recognition
          </p>
        </div>
        {isAnalyzing && (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-8 h-8 border-3 border-blue-400 border-t-transparent rounded-full"
          />
        )}
      </div>

      {analysisResults ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Primary Diagnosis */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg p-6 border border-white/20"
          >
            <h3 className="text-lg font-semibold text-gray-300 mb-4">
              Primary Diagnosis
            </h3>
            <div className="text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200 }}
                className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 mb-2"
              >
                {analysisResults.condition}
              </motion.div>
              <div className="flex items-center justify-center space-x-2 mb-4">
                <span className="text-gray-400">Confidence:</span>
                <span className="text-2xl font-bold text-emerald-400">
                  {(analysisResults.confidence * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Confidence Bar */}
              <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${analysisResults.confidence * 100}%` }}
                  transition={{ duration: 0.5 }}
                  className={`h-full rounded-full ${
                    analysisResults.confidence > 0.8 ? 'bg-emerald-500' :
                    analysisResults.confidence > 0.6 ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`}
                />
              </div>
            </div>
          </motion.div>

          {/* All Probabilities */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg p-6 border border-white/20"
          >
            <h3 className="text-lg font-semibold text-gray-300 mb-4">
              Differential Diagnosis
            </h3>
            <div className="space-y-3 max-h-64 overflow-y-auto custom-scrollbar">
              {Object.entries(analysisResults.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([condition, probability], index) => (
                  <motion.div
                    key={condition}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between"
                  >
                    <span className="text-sm text-gray-300 flex-1">{condition}</span>
                    <div className="flex items-center space-x-2 flex-1">
                      <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                          style={{ width: `${probability * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-12 text-right">
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </motion.div>
                ))}
            </div>
          </motion.div>

          {/* Feature Extraction Visualization */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-br from-emerald-500/20 to-teal-500/20 rounded-lg p-6 border border-white/20 lg:col-span-2"
          >
            <h3 className="text-lg font-semibold text-gray-300 mb-4">
              Extracted Features
            </h3>
            <div className="grid grid-cols-5 gap-4">
              {analysisResults.features.map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="text-center bg-black/30 rounded p-3"
                >
                  <p className="text-xs text-gray-400 mb-1">F{index + 1}</p>
                  <p className="text-lg font-bold text-white">
                    {feature.toFixed(3)}
                  </p>
                  <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
                    <div
                      className="h-full bg-teal-500 rounded-full"
                      style={{ width: `${Math.abs(feature) * 100}%` }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Model Information */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-br from-gray-500/20 to-slate-500/20 rounded-lg p-6 border border-white/20 lg:col-span-2"
          >
            <h3 className="text-lg font-semibold text-gray-300 mb-4">
              Model Architecture
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Input Layer</p>
                <p className="text-2xl font-bold text-blue-400">500</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Hidden Layers</p>
                <p className="text-2xl font-bold text-purple-400">256→128→64→32</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Output Classes</p>
                <p className="text-2xl font-bold text-emerald-400">10</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Activation</p>
                <p className="text-2xl font-bold text-pink-400">ReLU</p>
              </div>
            </div>
          </motion.div>
        </div>
      ) : (
        <div className="text-center py-12">
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center"
          >
            <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
            </svg>
          </motion.div>
          <p className="text-gray-400">
            Start recording to begin real-time ECG analysis
          </p>
        </div>
      )}
    </div>
  );
};

export default MLPAnalyzer;
