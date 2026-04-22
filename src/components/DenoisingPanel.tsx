import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PatientECGData } from '../utils/patientDataLoader';

interface DenoisingStatus {
  completed: boolean;
  isRunning: boolean;
  processedRecords: number;
  pythonAvailable: boolean;
  featurePreservationSummary?: {
    processed_records: number;
    processed_leads: number;
    mean_correlation: number;
    mean_energy_retention_pct: number;
    mean_peak_retention_pct: number;
    feature_preservation_pass_rate_pct: number;
  } | null;
}

interface DenoisingPanelProps {
  isOpen: boolean;
  selectedPatient: PatientECGData | null;
}

const DenoisingPanel: React.FC<DenoisingPanelProps> = ({ isOpen, selectedPatient }) => {
  const [status, setStatus] = useState<DenoisingStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [targetFs, setTargetFs] = useState(500);
  const [processId, setProcessId] = useState<string | null>(null);
  const [output, setOutput] = useState<string>('');
  const [error, setError] = useState<string>('');

  // Check status periodically
  useEffect(() => {
    if (!isOpen) return;

    const checkStatus = async () => {
      try {
        const response = await fetch('http://localhost:3001/api/denoise/status');
        const data = await response.json();
        setStatus(data);
      } catch (err) {
        console.error('Failed to fetch denoising status:', err);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, [isOpen]);

  // Poll output when process is running
  useEffect(() => {
    if (!processId || !status?.isRunning) return;

    const pollOutput = async () => {
      try {
        const response = await fetch(`http://localhost:3001/api/denoise/output/${processId}`);
        const data = await response.json();
        setOutput(data.output);
        setError(data.error);
        
        if (!data.running && data.exitCode !== null) {
          setProcessId(null);
        }
      } catch (err) {
        console.error('Failed to fetch output:', err);
      }
    };

    pollOutput();
    const interval = setInterval(pollOutput, 1000);
    return () => clearInterval(interval);
  }, [processId, status?.isRunning]);

  const startDenoising = async () => {
    if (!selectedPatient) {
      setError('Please select a patient first');
      return;
    }
    
    setLoading(true);
    setOutput('');
    setError('');
    
    try {
      const response = await fetch('http://localhost:3001/api/denoise/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          patientId: selectedPatient.patientId,
          targetFs 
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setProcessId(data.processId);
        setOutput(`Started denoising patient ${selectedPatient.patientId}...\n`);
      } else {
        setError(data.error || 'Failed to start processing');
      }
    } catch (err) {
      setError('Failed to connect to denoising server. Make sure it\'s running on port 3001.');
    } finally {
      setLoading(false);
    }
  };

  const stopDenoising = async () => {
    if (!processId) return;
    
    try {
      await fetch(`http://localhost:3001/api/denoise/stop/${processId}`, {
        method: 'POST'
      });
      setProcessId(null);
      setOutput(prev => prev + '\n[Process stopped by user]');
    } catch (err) {
      setError('Failed to stop process');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="border-t border-gray-300 pt-8 mt-8">
      <div className="flex items-center space-x-2 mb-4">
        <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        <h4 className="text-base font-bold text-gray-700">Signal Denoising</h4>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-white rounded-lg p-3 shadow-sm">
          <span className="text-sm text-gray-600">Python Status</span>
          <div className="flex items-center space-x-2 mt-1">
            <div className={`w-2 h-2 rounded-full ${status?.pythonAvailable ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-base font-bold text-gray-800">
              {status?.pythonAvailable ? 'Available' : 'Not Found'}
            </span>
          </div>
        </div>
        
        <div className="bg-white rounded-lg p-3 shadow-sm">
          <span className="text-sm text-gray-600">Processed Records</span>
          <span className="block text-base font-bold text-gray-800 mt-1">
            {status?.processedRecords || 0}
          </span>
        </div>
      </div>

      {status?.featurePreservationSummary && (
        <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-200 mb-4">
          <h5 className="text-sm font-bold text-emerald-900 mb-3">Feature Preservation (Raw vs Denoised)</h5>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-emerald-700">Correlation</p>
              <p className="font-bold text-emerald-900">{status.featurePreservationSummary.mean_correlation.toFixed(3)}</p>
            </div>
            <div>
              <p className="text-emerald-700">Energy Retention</p>
              <p className="font-bold text-emerald-900">{status.featurePreservationSummary.mean_energy_retention_pct.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-emerald-700">Peak Retention</p>
              <p className="font-bold text-emerald-900">{status.featurePreservationSummary.mean_peak_retention_pct.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-emerald-700">Pass Rate</p>
              <p className="font-bold text-emerald-900">{status.featurePreservationSummary.feature_preservation_pass_rate_pct.toFixed(1)}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Configuration */}
      {!status?.isRunning && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="space-y-3 mb-4"
        >
          {/* Selected Patient Info */}
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">
              Selected Patient
            </label>
            {selectedPatient ? (
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border-2 border-indigo-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-bold text-indigo-900">{selectedPatient.patientId}</p>
                    <p className="text-sm text-gray-600">
                      {selectedPatient.age}y • {selectedPatient.sex} • {Object.keys(selectedPatient.leads).length} leads
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-indigo-600 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
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
                  <span>Please select a patient from the dashboard first</span>
                </p>
              </div>
            )}
          </div>
          
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">
              Target Sampling Rate (Hz)
            </label>
            <select
              value={targetFs}
              onChange={(e) => setTargetFs(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            >
              <option value="100">100 Hz</option>
              <option value="250">250 Hz</option>
              <option value="500">500 Hz</option>
              <option value="1000">1000 Hz</option>
            </select>
          </div>
        </motion.div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2 mb-4">
        {!status?.isRunning ? (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={startDenoising}
            disabled={loading || !status?.pythonAvailable}
            className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-4 py-2.5 rounded-lg font-medium shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
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
                  <path d="M8 5v14l11-7z" />
                </svg>
                <span>Start Denoising</span>
              </span>
            )}
          </motion.button>
        ) : (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={stopDenoising}
            className="flex-1 bg-gradient-to-r from-red-600 to-orange-600 text-white px-4 py-2.5 rounded-lg font-medium shadow-lg hover:shadow-xl transition-all"
          >
            <span className="flex items-center justify-center space-x-2">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
              <span>Stop Processing</span>
            </span>
          </motion.button>
        )}
      </div>

      {/* Output Console */}
      {(output || error) && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-64 overflow-y-auto"
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
          {status?.isRunning && (
            <div className="text-yellow-400 animate-pulse mt-2">
              ● Processing...
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default DenoisingPanel;
