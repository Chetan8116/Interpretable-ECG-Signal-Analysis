import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PatientECGData } from '../utils/patientDataLoader';

interface FeatureExtractionPanelProps {
  isOpen: boolean;
  selectedPatient: PatientECGData | null;
}

const FeatureExtractionPanel: React.FC<FeatureExtractionPanelProps> = ({
  isOpen,
  selectedPatient
}) => {
  const [loading, setLoading] = useState(false);
  const [processId, setProcessId] = useState<string | null>(null);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [features, setFeatures] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);

  // Poll for output
  useEffect(() => {
    if (!processId || !status?.running) return;

    const pollOutput = async () => {
      try {
        const response = await fetch(`http://localhost:3001/api/features/status/${processId}`);
        const data = await response.json();
        
        if (data.output) {
          setOutput(data.output);
        }
        if (data.error) {
          setError(data.error);
        }
        
        if (!data.running) {
          setStatus({ ...status, running: false });
          // Load results
          if (selectedPatient && data.exitCode === 0) {
            loadFeatures();
          }
        }
      } catch (err) {
        console.error('Failed to fetch process status:', err);
      }
    };

    pollOutput();
    const interval = setInterval(pollOutput, 1000);
    return () => clearInterval(interval);
  }, [processId, status?.running, selectedPatient]);

  const startExtraction = async () => {
    if (!selectedPatient) {
      setError('Please select a patient first');
      return;
    }
    
    setLoading(true);
    setOutput('');
    setError('');
    setFeatures(null);
    
    try {
      const response = await fetch('http://localhost:3001/api/features/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          patientId: selectedPatient.patientId
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setProcessId(data.processId);
        setStatus({ running: true });
        setOutput(`Started feature extraction for patient ${selectedPatient.patientId}...\n`);
      } else {
        setError(data.error || 'Failed to start feature extraction');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure backend is running on port 3001.');
    } finally {
      setLoading(false);
    }
  };

  const loadFeatures = async () => {
    if (!selectedPatient) return;
    
    try {
      const response = await fetch(`http://localhost:3001/api/features/results/${selectedPatient.patientId}`);
      const data = await response.json();
      
      if (data.success) {
        setFeatures(data.features);
      }
    } catch (err) {
      console.error('Failed to load features:', err);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="border-t border-gray-300 pt-4 mt-4">
      <div className="flex items-center space-x-3 mb-4">
        <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <h3 className="text-lg font-bold text-gray-800">Feature Extraction</h3>
      </div>

      {/* Patient Info */}
      {!status?.running && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="space-y-3 mb-4"
        >
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">
              Selected Patient
            </label>
            {selectedPatient ? (
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4 border-2 border-purple-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-bold text-purple-900">{selectedPatient.patientId}</p>
                    <p className="text-sm text-gray-600">
                      {selectedPatient.age}y • {selectedPatient.sex} • {Object.keys(selectedPatient.leads).length} leads
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                <p className="text-sm text-yellow-800 flex items-center space-x-2">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span>Please select a patient and denoise first</span>
                </p>
              </div>
            )}
          </div>

          <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
            <p className="text-sm text-blue-800">
              <strong>What will be extracted:</strong>
            </p>
            <ul className="text-xs text-blue-700 mt-2 space-y-1 ml-4">
              <li>• Window features (variance, skewness, kurtosis, entropy)</li>
              <li>• QRS detection maps using logistic regression</li>
              <li>• R-peak detection and refinement</li>
              <li>• Enhanced signal with Hilbert + morphology</li>
              <li>• Heart rate estimation per lead</li>
            </ul>
          </div>
        </motion.div>
      )}

      {/* Action Button */}
      <div className="flex space-x-3 mb-4">
        {!status?.running ? (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={startExtraction}
            disabled={loading || !selectedPatient}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 text-white px-4 py-2.5 rounded-lg font-medium shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center justify-center space-x-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Starting...</span>
              </span>
            ) : (
              <span className="flex items-center justify-center space-x-2">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Extract Features</span>
              </span>
            )}
          </motion.button>
        ) : (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="flex-1 bg-gray-500 text-white px-4 py-2.5 rounded-lg font-medium shadow-lg cursor-not-allowed"
            disabled
          >
            <span className="flex items-center justify-center space-x-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>Processing...</span>
            </span>
          </motion.button>
        )}
      </div>

      {/* Output Console */}
      {(output || error) && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-64 overflow-y-auto mb-4"
        >
          {output && (
            <div className="text-green-400 whitespace-pre-wrap mb-2">
              {output}
            </div>
          )}
          {error && (
            <div className="text-red-400 whitespace-pre-wrap">
              {error}
            </div>
          )}
          {status?.running && (
            <div className="text-yellow-400 animate-pulse mt-2">
              * Processing features...
            </div>
          )}
        </motion.div>
      )}

      {/* Results Display */}
      {features && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-4 border-2 border-purple-200"
        >
          <h4 className="font-bold text-purple-900 mb-3 flex items-center space-x-2">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
            </svg>
            <span>Extracted Features</span>
          </h4>
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(features).map(([lead, data]: [string, any]) => (
              <div key={lead} className="bg-white rounded-lg p-3 shadow-sm">
                <p className="text-xs font-semibold text-purple-700 uppercase">{lead}</p>
                <div className="mt-2 space-y-1">
                  <p className="text-sm text-gray-700">
                    <span className="font-medium">R-peaks:</span> {data.num_peaks}
                  </p>
                  <p className="text-sm text-gray-700">
                    <span className="font-medium">HR:</span> {data.heart_rate} bpm
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default FeatureExtractionPanel;
