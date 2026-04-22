import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';

interface SHAPFeature {
  feature: string;
  lead: string;
  feature_type: string;
  shap_value: number;
  feature_value: number;
}

interface SHAPResult {
  ecg_id: number;
  true_label: string;
  predicted_class: string;
  class_probabilities: Record<string, number>;
  top_shap_features: SHAPFeature[];
  all_shap_features: SHAPFeature[];
  lead_total_shap: Record<string, number>;
  lead_signed_shap: Record<string, number>;
  shap_base_value: number;
}

interface LeadSHAPSummary {
  lead: string;
  totalShap: number;
  signedShap: number;
  topFeatures: string[];
  featureDetails: { feature: string; shap: number }[];
}

const SHAP_SERVER = 'http://localhost:5101';

const LeadWiseSHAP: React.FC = () => {
  const { selectedPatient } = useECGStore();
  const [shapResult, setShapResult] = useState<SHAPResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'checking' | 'ready' | 'offline'>('checking');
  const [isExpanded, setIsExpanded] = useState(true);

  // Check server on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await fetch(`${SHAP_SERVER}/api/shap/status`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
          setServerStatus('ready');
        } else {
          setServerStatus('offline');
        }
      } catch {
        setServerStatus('offline');
      }
    };
    checkServer();
  }, []);

  // Fetch real SHAP data when patient changes
  useEffect(() => {
    if (!selectedPatient || serverStatus !== 'ready') {
      setShapResult(null);
      return;
    }

    const fetchSHAP = async () => {
      setLoading(true);
      setError(null);
      try {
        const numericId = selectedPatient.patientId.replace(/\D/g, '');
        const res = await fetch(`${SHAP_SERVER}/api/shap/${numericId}`);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error || `Server returned ${res.status}`);
        }
        const data: SHAPResult = await res.json();
        setShapResult(data);
      } catch (e: any) {
        setError(e.message || 'Failed to fetch SHAP data');
        setShapResult(null);
      } finally {
        setLoading(false);
      }
    };

    fetchSHAP();
  }, [selectedPatient, serverStatus]);

  // Derive lead summaries from real SHAP result
  const leadSummaries: LeadSHAPSummary[] = shapResult
    ? (() => {
        const byLead: Record<string, { feature: string; shap: number }[]> = {};
        shapResult.all_shap_features.forEach(f => {
          if (!byLead[f.lead]) byLead[f.lead] = [];
          byLead[f.lead].push({ feature: f.feature_type, shap: f.shap_value });
        });
        return Object.entries(shapResult.lead_total_shap)
          .map(([lead, totalShap]) => {
            const details = (byLead[lead] || []).sort(
              (a, b) => Math.abs(b.shap) - Math.abs(a.shap)
            );
            return {
              lead,
              totalShap,
              signedShap: shapResult.lead_signed_shap[lead] ?? 0,
              topFeatures: details.slice(0, 3).map(f => f.feature),
              featureDetails: details,
            };
          })
          .sort((a, b) => b.totalShap - a.totalShap);
      })()
    : [];

  if (!selectedPatient) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">Select a patient to see lead-wise SHAP analysis</p>
      </div>
    );
  }

  if (serverStatus === 'offline') {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-5">
          <h3 className="font-semibold text-amber-800 mb-2">⚠️ SHAP Server Offline</h3>
          <p className="text-sm text-amber-700 mb-3">Start the Python SHAP server to compute real lead-wise SHAP values:</p>
          <code className="block bg-white border border-amber-200 rounded-lg px-4 py-2 text-sm font-mono text-gray-800 mb-2">
            pip install flask flask-cors shap scikit-learn pandas numpy
          </code>
          <code className="block bg-white border border-amber-200 rounded-lg px-4 py-2 text-sm font-mono text-gray-800">
            python server/shap_server.py
          </code>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
          <p className="text-gray-600">Computing SHAP by lead…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-red-700 font-semibold">Error: {error}</p>
        </div>
      </div>
    );
  }

  if (leadSummaries.length === 0) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">No lead-wise SHAP data available for this patient</p>
      </div>
    );
  }

  const maxTotalShap = Math.max(...leadSummaries.map(s => s.totalShap));

  return (
    <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Lead-Wise SHAP Aggregation</h2>
            <p className="text-sm text-gray-500">Clinical Interpretation by Lead</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className="px-4 py-2 bg-indigo-100 text-indigo-700 rounded-full text-sm font-semibold flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            LIVE
          </div>
          <motion.button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <motion.svg
              className="w-6 h-6 text-gray-700"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </motion.svg>
          </motion.button>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
      {/* Prediction Summary */}
      {shapResult && (
        <div className="mb-6 flex flex-wrap gap-3 items-center">
          <span className="px-3 py-1 bg-indigo-600 text-white rounded-full text-sm font-semibold">
            Predicted: {shapResult.predicted_class}
          </span>
          {shapResult.true_label && (
            <span className={`px-3 py-1 rounded-full text-sm font-semibold ${shapResult.true_label === shapResult.predicted_class ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
              True: {shapResult.true_label}
            </span>
          )}
          <span className="text-xs text-gray-500">SHAP base value: {shapResult.shap_base_value.toFixed(4)}</span>
        </div>
      )}

      {/* Description */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-4 mb-6">
        <p className="text-sm text-gray-700">
          <strong>Lead-wise aggregation</strong> groups real SHAP values by ECG lead, showing which leads contributed most to the diagnosis. 
          This mirrors how cardiologists analyze ECGs — thinking in terms of anatomical regions represented by each lead.
          <span className="ml-2 text-red-600 font-semibold">Red = pushes toward predicted class.</span>
          <span className="ml-2 text-blue-600 font-semibold">Blue = pushes away.</span>
        </p>
      </div>

      {/* Lead Summary Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b-2 border-gray-200">
              <th className="px-4 py-3 text-left font-semibold text-gray-700 bg-gradient-to-r from-indigo-50 to-indigo-100 rounded-tl-xl">
                Rank
              </th>
              <th className="px-4 py-3 text-left font-semibold text-gray-700 bg-gradient-to-r from-indigo-50 to-indigo-100">
                Lead
              </th>
              <th className="px-4 py-3 text-center font-semibold text-gray-700 bg-gradient-to-r from-indigo-50 to-indigo-100">
                Total SHAP
              </th>
              <th className="px-4 py-3 text-left font-semibold text-gray-700 bg-gradient-to-r from-indigo-50 to-indigo-100 rounded-tr-xl">
                Top Features
              </th>
            </tr>
          </thead>
          <tbody>
            {leadSummaries.map((summary, index) => {
              const percentage = (summary.totalShap / maxTotalShap) * 100;
              
              return (
                <motion.tr
                  key={summary.lead}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.08 }}
                  className={`border-b border-gray-200 hover:bg-indigo-50/50 transition-colors cursor-pointer ${
                    index === leadSummaries.length - 1 ? 'border-b-0' : ''
                  }`}
                >
                  <td className="px-4 py-4 text-center font-semibold text-gray-600">
                    #{index + 1}
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-indigo-400 to-purple-600 flex items-center justify-center shadow-md">
                        <span className="text-white font-bold text-sm">{summary.lead}</span>
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">{summary.lead}</p>
                        <p className="text-xs text-gray-500">{summary.featureDetails.length} features</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex flex-col items-center">
                      <span className={`text-2xl font-bold mb-2 ${summary.signedShap >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                        {summary.signedShap >= 0 ? '+' : ''}{summary.signedShap.toFixed(3)}
                      </span>
                      <div className="text-xs text-gray-400 mb-1">|{summary.totalShap.toFixed(3)}| total</div>
                      <div className="w-32 bg-gray-200 rounded-full h-2 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${percentage}%` }}
                          transition={{ duration: 1, delay: 0.3 + index * 0.08 }}
                          className={`h-full rounded-full ${summary.signedShap >= 0 ? 'bg-gradient-to-r from-red-400 to-red-600' : 'bg-gradient-to-r from-blue-400 to-blue-600'}`}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex flex-wrap gap-2">
                      {summary.topFeatures.map((feature, idx) => (
                        <motion.span
                          key={feature}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: 0.4 + index * 0.08 + idx * 0.05 }}
                          className={`px-3 py-1 rounded-full text-xs font-semibold ${
                            idx === 0 
                              ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white' 
                              : idx === 1
                              ? 'bg-indigo-100 text-indigo-700'
                              : 'bg-gray-100 text-gray-700'
                          }`}
                        >
                          {feature}
                        </motion.span>
                      ))}
                    </div>
                    <div className="mt-2 text-xs text-gray-500">
                      {summary.featureDetails.slice(0, 3).map((f, idx) => (
                        <span key={idx} className="mr-3">
                          {f.feature}: {f.shap.toFixed(2)}
                        </span>
                      ))}
                    </div>
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Clinical Interpretation */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Clinical Interpretation</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {leadSummaries.slice(0, 3).map((summary, index) => (
            <motion.div
              key={summary.lead}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + index * 0.1 }}
              className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-4"
            >
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                  <span className="text-white font-bold text-sm">{summary.lead}</span>
                </div>
                <span className="font-semibold text-gray-800">Lead {summary.lead}</span>
              </div>
              <p className="text-sm text-gray-700 mb-2">
                <strong>Signed SHAP:</strong> {summary.signedShap >= 0 ? '+' : ''}{summary.signedShap.toFixed(3)}
              </p>
              <p className="text-xs text-gray-600">
                <strong>Key abnormalities:</strong> {summary.topFeatures.join(', ')}
              </p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Lead Groups Info */}
      <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-2">📍 Lead Groups & Anatomy</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-600">
          <div>
            <strong>Inferior:</strong> II, III, aVF (bottom of heart)
          </div>
          <div>
            <strong>Lateral:</strong> I, aVL, V5, V6 (left side)
          </div>
          <div>
            <strong>Anterior:</strong> V1-V4 (front of heart)
          </div>
        </div>
      </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LeadWiseSHAP;
