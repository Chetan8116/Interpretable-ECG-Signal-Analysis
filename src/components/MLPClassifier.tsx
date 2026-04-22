import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  probabilities: { [key: string]: number };
  feature_vector_size: number;
}

const MLPClassifier: React.FC = () => {
  const { selectedPatient } = useECGStore();
  const [features, setFeatures] = useState<number[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const extractAndPredict = async () => {
      if (!selectedPatient) {
        setPrediction(null);
        return;
      }

      setLoading(true);
      try {
        // Extract features from PTB-XL records
        const response = await fetch('/ptbxl_records.json');
        const records = await response.json();
        
        const numericId = parseInt(selectedPatient.patientId.replace('PTB', ''));
        const patientRecord = records.find((r: any) => r.ecg_id === numericId);
        
        if (!patientRecord || !patientRecord.leads) {
          setPrediction(null);
          setLoading(false);
          return;
        }

        // Extract clinical features from all 12 leads
        const leadNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
        const featureKeys = [
          'P_duration_mean',
          'P_amplitude_mean',
          'P_area_mean',
          'PR_interval_mean',
          'QRS_width_mean',
          'R_amplitude_mean',
          'QRS_area_mean',
          'ST_elevation_mean',
          'ST_slope_mean',
          'T_amplitude_mean',
          'T_area_mean',
          'QT_interval_mean',
          'Heart_rate_mean',
          'RR_interval_mean',
          'QTc_mean',
          'P_axis_mean',
          'QRS_axis_mean'
        ];

        const featureVector: number[] = [];
        
        // Flatten: for each lead, extract all features (17 features × 12 leads = 204 dimensions)
        leadNames.forEach(leadName => {
          const leadData = patientRecord.leads[leadName];
          if (leadData && leadData.clinical_features) {
            const cf = leadData.clinical_features;
            featureKeys.forEach(key => {
              featureVector.push(cf[key] || 0);
            });
          } else {
            // Fill with zeros if lead data missing
            featureKeys.forEach(() => featureVector.push(0));
          }
        });

        setFeatures(featureVector);

        // Simple rule-based prediction (since we can't run Python model in browser)
        // In production, you'd call a backend API with the trained model
        const prediction = makePrediction(featureVector);
        setPrediction(prediction);
        
      } catch (error) {
        console.error('Error extracting features:', error);
        setPrediction(null);
      } finally {
        setLoading(false);
      }
    };

    extractAndPredict();
  }, [selectedPatient]);

  // Simple rule-based prediction mimicking MLP behavior
  const makePrediction = (features: number[]): PredictionResult => {
    // Extract key features for decision
    const avgSTElevation = features.filter((_, i) => i % 17 === 7).reduce((a, b) => a + b, 0) / 12;
    const avgHeartRate = features.filter((_, i) => i % 17 === 12).reduce((a, b) => a + b, 0) / 12;
    const avgQRSWidth = features.filter((_, i) => i % 17 === 4).reduce((a, b) => a + b, 0) / 12;
    
    let predicted_class = 'NORM';
    const probabilities: { [key: string]: number } = {
      'NORM': 0.85,
      'MI': 0.10,
      'OTHER': 0.05
    };

    // Simple rules
    if (Math.abs(avgSTElevation) > 0.1) {
      predicted_class = 'MI';
      probabilities['MI'] = 0.75;
      probabilities['NORM'] = 0.15;
      probabilities['OTHER'] = 0.10;
    } else if (avgHeartRate < 50 || avgHeartRate > 120 || avgQRSWidth > 0.12) {
      predicted_class = 'OTHER';
      probabilities['OTHER'] = 0.65;
      probabilities['NORM'] = 0.25;
      probabilities['MI'] = 0.10;
    }

    return {
      predicted_class,
      confidence: probabilities[predicted_class],
      probabilities,
      feature_vector_size: features.length
    };
  };

  if (!selectedPatient) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">Select a patient to see MLP classification</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
          <p className="text-gray-600">Extracting features and classifying...</p>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">Unable to generate prediction for this patient</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">MLP Classification</h2>
            <p className="text-sm text-gray-500">Neural Network Prediction</p>
          </div>
        </div>
        <div className="px-4 py-2 bg-purple-100 text-purple-700 rounded-full text-sm font-semibold">
          Step 4: Prediction
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Features */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="w-2 h-2 bg-blue-500 rounded-full mr-2 animate-pulse"></span>
            Input Features
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Feature Vector Size</span>
              <span className="text-xl font-bold text-gray-800">{prediction.feature_vector_size}</span>
            </div>
            <div className="text-xs text-gray-500">
              17 clinical features × 12 leads = 204 dimensions
            </div>
            <div className="bg-white/50 rounded-lg p-3">
              <p className="text-xs text-gray-600 mb-2">Sample features (first 12):</p>
              <div className="grid grid-cols-4 gap-2">
                {features.slice(0, 12).map((val, idx) => (
                  <div key={idx} className="text-center bg-white rounded px-2 py-1">
                    <span className="text-xs font-mono text-gray-700">{val.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Prediction Result */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-purple-50 to-pink-100 rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="w-2 h-2 bg-purple-500 rounded-full mr-2 animate-pulse"></span>
            Predicted Class
          </h3>
          <div className="text-center mb-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 200 }}
              className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-500 mb-2"
            >
              {prediction.predicted_class}
            </motion.div>
            <div className="flex items-center justify-center space-x-2 mb-4">
              <span className="text-gray-600 text-sm">Confidence:</span>
              <span className="text-2xl font-bold text-purple-600">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${prediction.confidence * 100}%` }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className={`h-full rounded-full ${
                  prediction.confidence > 0.7 ? 'bg-gradient-to-r from-emerald-500 to-emerald-600' :
                  prediction.confidence > 0.5 ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' :
                  'bg-gradient-to-r from-red-500 to-red-600'
                }`}
              />
            </div>
          </div>
        </motion.div>

        {/* Class Probabilities */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 bg-gradient-to-br from-emerald-50 to-teal-100 rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
            Class Probabilities
          </h3>
          <div className="space-y-3">
            {Object.entries(prediction.probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([className, probability], index) => (
                <motion.div
                  key={className}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="flex items-center space-x-4"
                >
                  <div className="w-24 text-right">
                    <span className="font-semibold text-gray-800">{className}</span>
                  </div>
                  <div className="flex-1">
                    <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${probability * 100}%` }}
                        transition={{ duration: 1, delay: 0.4 + index * 0.1 }}
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                      />
                    </div>
                  </div>
                  <div className="w-20 text-right">
                    <span className="text-lg font-bold text-gray-800">
                      {(probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              ))}
          </div>
        </motion.div>

        {/* Pipeline Visualization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Complete Pipeline: ECG → Features → Prediction
          </h3>
          <div className="flex items-center justify-center space-x-4">
            <div className="text-center">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center mb-2 shadow-lg">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <p className="text-xs font-semibold text-gray-700">ECG Signal</p>
              <p className="text-xs text-gray-500">12 Leads</p>
            </div>
            
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            
            <div className="text-center">
              <div className="w-24 h-24 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-xl flex items-center justify-center mb-2 shadow-lg">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-xs font-semibold text-gray-700">Feature Table</p>
              <p className="text-xs text-gray-500">204 features</p>
            </div>
            
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            
            <div className="text-center">
              <div className="w-24 h-24 bg-gradient-to-br from-purple-400 to-pink-600 rounded-xl flex items-center justify-center mb-2 shadow-lg">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <p className="text-xs font-semibold text-gray-700">MLP Classifier</p>
              <p className="text-xs text-gray-500">204→128→64→32→3</p>
            </div>
            
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            
            <div className="text-center">
              <div className="w-24 h-24 bg-gradient-to-br from-pink-400 to-red-600 rounded-xl flex items-center justify-center mb-2 shadow-lg">
                <span className="text-2xl font-bold text-white">{prediction.predicted_class}</span>
              </div>
              <p className="text-xs font-semibold text-gray-700">Prediction</p>
              <p className="text-xs text-gray-500">{(prediction.confidence * 100).toFixed(0)}% conf</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Note */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          ℹ️ This is a demonstration using rule-based classification. In production, predictions would be made using the trained MLP model via a Python backend API.
        </p>
      </div>
    </div>
  );
};

export default MLPClassifier;
