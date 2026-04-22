import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';

const SHAP_SERVER = 'http://localhost:5101';

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
  shap_base_value: number;
}

type ServerStatus = 'checking' | 'ready' | 'offline';

const SHAPExplainability: React.FC = () => {
  const { selectedPatient } = useECGStore();
  const [shapResult, setShapResult] = useState<SHAPResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>('checking');
  const [isExpanded, setIsExpanded] = useState(true);

  /** Poll the SHAP server status once on mount */
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const r = await fetch(`${SHAP_SERVER}/api/shap/status`, { signal: AbortSignal.timeout(3000) });
        if (cancelled) return;
        const json = await r.json();
        setServerStatus(json.ready ? 'ready' : 'offline');
      } catch {
        if (!cancelled) setServerStatus('offline');
      }
    };
    check();
    return () => { cancelled = true; };
  }, []);

  /** Fetch real SHAP values whenever the patient changes */
  useEffect(() => {
    if (!selectedPatient) {
      setShapResult(null);
      setError(null);
      return;
    }
    if (serverStatus === 'offline') return;

    const numericId = parseInt(selectedPatient.patientId.replace(/\D/g, ''), 10);
    if (isNaN(numericId)) return;

    let cancelled = false;
    const fetchSHAP = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${SHAP_SERVER}/api/shap/${numericId}`);
        if (cancelled) return;
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || `HTTP ${res.status}`);
        }
        const data: SHAPResult = await res.json();
        setShapResult(data);
      } catch (e: any) {
        if (!cancelled) {
          setError(e.message ?? 'Unknown error');
          setShapResult(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchSHAP();
    return () => { cancelled = true; };
  }, [selectedPatient, serverStatus]);

  /* ── Render helpers ──────────────────────────────────────────────────────── */

  if (!selectedPatient) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">Select a patient to see SHAP explainability</p>
      </div>
    );
  }

  if (serverStatus === 'offline') {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
          <span className="font-semibold text-red-600">SHAP server offline</span>
        </div>
        <p className="text-sm text-gray-600 mb-2">
          Start the Python SHAP server to enable real explainability:
        </p>
        <pre className="bg-gray-900 text-green-400 rounded-lg p-4 text-xs overflow-x-auto">
          pip install flask flask-cors shap scikit-learn{'\n'}
          python server/shap_server.py
        </pre>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-orange-200 border-t-orange-500" />
          <p className="text-gray-600 font-medium">Computing real SHAP values…</p>
          <p className="text-xs text-gray-400">Using KernelExplainer on trained MLP model</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-red-600 font-semibold">Error: {error}</p>
        <p className="text-sm text-gray-500 mt-1">Patient may not be in the feature dataset.</p>
      </div>
    );
  }

  if (!shapResult) return null;

  const features = shapResult.top_shap_features;
  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 1e-9);
  const sortedClasses = Object.entries(shapResult.class_probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
      {/* ── Header ────────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">SHAP Explainability</h2>
            <p className="text-sm text-gray-500">Real KernelSHAP · trained MLP model</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1 px-3 py-1 rounded-full bg-green-100 text-green-700 text-xs font-semibold">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>LIVE</span>
          </div>
          <motion.button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
            whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          >
            <motion.svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"
              animate={{ rotate: isExpanded ? 180 : 0 }} transition={{ duration: 0.3 }}>
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
            {/* ── Prediction summary ──────────────────────────────────────────── */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-4">
                <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Predicted Diagnosis</p>
                <div className="flex items-center space-x-3">
                  <span className="text-3xl font-extrabold text-orange-600">{shapResult.predicted_class}</span>
                  {shapResult.true_label && shapResult.true_label !== 'UNKNOWN' && (
                    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      shapResult.predicted_class === shapResult.true_label
                        ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                      True: {shapResult.true_label}
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">Base value: {shapResult.shap_base_value.toFixed(4)}</p>
              </div>
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Class Probabilities</p>
                <div className="space-y-1">
                  {sortedClasses.map(([cls, prob]) => (
                    <div key={cls} className="flex items-center space-x-2">
                      <span className="text-xs font-medium w-12 text-gray-700">{cls}</span>
                      <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${prob * 100}%` }}
                          transition={{ duration: 0.8 }}
                          className={`h-full rounded-full ${
                            cls === shapResult.predicted_class
                              ? 'bg-gradient-to-r from-orange-400 to-red-500'
                              : 'bg-gray-300'
                          }`}
                        />
                      </div>
                      <span className="text-xs text-gray-600 w-10 text-right">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ── Description ─────────────────────────────────────────────────── */}
            <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-4 mb-6">
              <p className="text-sm text-gray-700">
                <strong>SHAP (SHapley Additive exPlanations)</strong> — computed by{' '}
                <code className="bg-orange-100 px-1 rounded text-xs">KernelExplainer</code> on the
                trained MLP model using 170 real clinical ECG features. Positive values push toward
                the predicted class; negative values push away.
              </p>
            </div>

            {/* ── Top-20 feature table ─────────────────────────────────────────── */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-gray-200">
                    {['Rank', 'Feature', 'Lead', 'SHAP Value', 'Feature Value', 'Contribution'].map((h, i) => (
                      <th key={i} className={`px-4 py-3 text-left font-semibold text-gray-700 bg-gradient-to-r from-orange-50 to-orange-100 ${i === 0 ? 'rounded-tl-xl' : ''} ${i === 5 ? 'rounded-tr-xl' : ''}`}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {features.map((feat, idx) => {
                    const pct = (Math.abs(feat.shap_value) / maxAbs) * 100;
                    const pos = feat.shap_value > 0;
                    return (
                      <motion.tr
                        key={feat.feature}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.04 }}
                        className="border-b border-gray-100 hover:bg-orange-50/40 transition-colors"
                      >
                        <td className="px-4 py-2.5 text-center font-semibold text-gray-500">#{idx + 1}</td>
                        <td className="px-4 py-2.5 font-mono text-xs text-gray-800">{feat.feature}</td>
                        <td className="px-4 py-2.5">
                          <span className="px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 text-xs font-semibold">
                            {feat.lead}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-center">
                          <span className={`font-bold ${pos ? 'text-red-600' : 'text-blue-600'}`}>
                            {pos ? '+' : ''}{feat.shap_value.toFixed(4)}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-center text-gray-600">
                          {feat.feature_value.toFixed(3)}
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center space-x-2">
                            <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${pct}%` }}
                                transition={{ duration: 0.7, delay: 0.1 + idx * 0.04 }}
                                className={`h-full rounded-full ${
                                  pos
                                    ? 'bg-gradient-to-r from-red-400 to-red-600'
                                    : 'bg-gradient-to-r from-blue-400 to-blue-600'
                                }`}
                              />
                            </div>
                            <span className="text-xs text-gray-500 w-10 text-right">{pct.toFixed(0)}%</span>
                          </div>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* ── Legend ──────────────────────────────────────────────────────── */}
            <div className="mt-6 pt-4 border-t border-gray-200 flex flex-wrap gap-4 text-xs text-gray-600">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded bg-gradient-to-r from-red-400 to-red-600" />
                <span>Positive → pushes toward predicted class</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded bg-gradient-to-r from-blue-400 to-blue-600" />
                <span>Negative → pushes away from predicted class</span>
              </div>
              <div className="ml-auto text-gray-400">Top 20 of 170 features shown</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SHAPExplainability;
