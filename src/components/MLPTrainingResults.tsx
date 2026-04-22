import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface MLPResults {
  accuracy: number;
  classes: string[];
  confusion_matrix: number[][];
  test_samples: number;
  architecture: {
    input_size: number;
    hidden_layers: number[];
    output_size: number;
    activation: string;
    solver: string;
  };
  training_info: {
    iterations: number;
    final_loss: number;
  };
  per_class_performance: {
    [key: string]: {
      accuracy: number;
      samples: number;
      present_in_test: boolean;
    };
  };
}

const MLPTrainingResults: React.FC = () => {
  const [results, setResults] = useState<MLPResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadResults = async () => {
      try {
        const response = await fetch('/mlp_results.json');
        const data = await response.json();
        setResults(data);
      } catch (error) {
        console.error('Error loading MLP results:', error);
      } finally {
        setLoading(false);
      }
    };

    loadResults();
  }, []);

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="text-gray-600">Loading MLP Training Results...</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">No training results available</p>
      </div>
    );
  }

  const accuracyPercent = results.accuracy * 100;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-6">
        <div className="flex items-center space-x-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">MLP Classification Results</h2>
            <p className="text-sm text-gray-500">Neural Network Training Performance</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Overall Accuracy Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8"
        >
          <h3 className="text-lg font-semibold text-gray-700 mb-6 flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
            Overall Accuracy
          </h3>
          <div className="flex items-center justify-center">
            <div className="relative w-48 h-48">
              {/* Background Circle */}
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="96"
                  cy="96"
                  r="88"
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="none"
                  className="text-gray-200"
                />
                {/* Progress Circle */}
                <motion.circle
                  cx="96"
                  cy="96"
                  r="88"
                  stroke="url(#gradient)"
                  strokeWidth="12"
                  fill="none"
                  strokeLinecap="round"
                  initial={{ strokeDashoffset: 553 }}
                  animate={{ strokeDashoffset: 553 - (553 * accuracyPercent) / 100 }}
                  transition={{ duration: 1.5, ease: 'easeOut' }}
                  style={{
                    strokeDasharray: 553,
                  }}
                />
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#10b981" />
                    <stop offset="100%" stopColor="#3b82f6" />
                  </linearGradient>
                </defs>
              </svg>
              {/* Center Text */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.5, type: 'spring' }}
                  className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-500 to-blue-500"
                >
                  {accuracyPercent.toFixed(0)}%
                </motion.div>
                <p className="text-sm text-gray-500 mt-1">{results.test_samples} samples</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Training Info Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8"
        >
          <h3 className="text-lg font-semibold text-gray-700 mb-6 flex items-center">
            <span className="w-2 h-2 bg-blue-500 rounded-full mr-2 animate-pulse"></span>
            Training Metrics
          </h3>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-600">Iterations</span>
                <span className="text-2xl font-bold text-gray-800">{results.training_info.iterations}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 1, delay: 0.3 }}
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-600">Final Loss</span>
                <span className="text-2xl font-bold text-gray-800">
                  {results.training_info.final_loss.toFixed(4)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(1 - results.training_info.final_loss) * 100}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                  className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full"
                />
              </div>
            </div>

            <div className="pt-4 border-t border-gray-200">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Convergence</span>
                <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-sm font-semibold">
                  Optimal
                </span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Per-Class Performance */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8"
        >
          <h3 className="text-lg font-semibold text-gray-700 mb-6 flex items-center">
            <span className="w-2 h-2 bg-pink-500 rounded-full mr-2 animate-pulse"></span>
            Per-Class Performance
          </h3>
          <div className="space-y-4">
            {results.classes.map((className, index) => {
              const perf = results.per_class_performance[className];
              const accuracyPercent = perf.accuracy * 100;
              
              return (
                <motion.div
                  key={className}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  className={`p-4 rounded-xl ${
                    perf.present_in_test ? 'bg-gradient-to-r from-blue-50 to-purple-50' : 'bg-gray-50'
                  }`}
                >
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-gray-800">{className}</span>
                      {!perf.present_in_test && (
                        <span className="px-2 py-0.5 bg-gray-300 text-gray-600 rounded text-xs">
                          Not in test set
                        </span>
                      )}
                    </div>
                    <span className="text-sm text-gray-600">{perf.samples} samples</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="flex-1">
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${accuracyPercent}%` }}
                          transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                          className={`h-full rounded-full ${
                            accuracyPercent >= 90 ? 'bg-gradient-to-r from-emerald-500 to-emerald-600' :
                            accuracyPercent >= 70 ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' :
                            accuracyPercent > 0 ? 'bg-gradient-to-r from-red-500 to-red-600' :
                            'bg-gray-400'
                          }`}
                        />
                      </div>
                    </div>
                    <span className="text-lg font-bold text-gray-800 w-16 text-right">
                      {perf.present_in_test ? `${accuracyPercent.toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default MLPTrainingResults;
