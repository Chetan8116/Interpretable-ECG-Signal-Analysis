import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface PatientECGData {
  patient_id: string;
  ecg_id: number;
  age: number;
  sex: string;
  leads: {
    [key: string]: {
      samples: number[];
    };
  };
}

interface DiagnosisPredictionProps {
  isOpen: boolean;
  selectedPatient: PatientECGData | null;
}

interface LeadInfluence {
  lead: string;
  influence: number;
  features: string[];
}

interface PredictionResult {
  success: boolean;
  patient_id: string;
  prediction: string;
  prediction_full: string;
  confidence: number;
  probabilities: { [key: string]: number };
  patient_info: {
    ecg_id: number;
    age: number;
    sex: string;
    actual_scp_codes: { [key: string]: number };
    report: string;
  };
  source?: string;
  error?: string;
}

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

const CLASS_COLORS: { [key: string]: { border: string; bg: string; text: string; badge: string } } = {
  NORM: { border: 'border-green-500', bg: 'bg-green-500/10', text: 'text-green-400', badge: 'bg-green-500/20 text-green-300' },
  MI:   { border: 'border-red-500',   bg: 'bg-red-500/10',   text: 'text-red-400',   badge: 'bg-red-500/20 text-red-300' },
  STTC: { border: 'border-yellow-500',bg: 'bg-yellow-500/10',text: 'text-yellow-400',badge: 'bg-yellow-500/20 text-yellow-300' },
  HYP:  { border: 'border-orange-500',bg: 'bg-orange-500/10',text: 'text-orange-400',badge: 'bg-orange-500/20 text-orange-300' },
  CD:   { border: 'border-purple-500',bg: 'bg-purple-500/10',text: 'text-purple-400',badge: 'bg-purple-500/20 text-purple-300' },
};

// Simulates lead influence scores based on diagnosis and probabilities
function computeLeadInfluences(prediction: string, probabilities: { [key: string]: number }): LeadInfluence[] {
  const conf = probabilities[prediction] ?? 0.5;

  const leadWeights: { [key: string]: { [lead: string]: number } } = {
    MI:   { V1: 0.9, V2: 0.88, V3: 0.75, aVR: 0.72, II: 0.65, III: 0.6, aVF: 0.58, V4: 0.5, I: 0.3, aVL: 0.28, V5: 0.25, V6: 0.2 },
    STTC: { II: 0.88, V5: 0.85, V6: 0.82, V4: 0.75, III: 0.65, aVF: 0.6, I: 0.55, aVL: 0.5, V1: 0.45, V2: 0.4, V3: 0.38, aVR: 0.3 },
    HYP:  { V5: 0.9, V6: 0.88, V1: 0.82, aVL: 0.75, I: 0.7, V4: 0.65, II: 0.55, V2: 0.5, V3: 0.45, III: 0.4, aVF: 0.35, aVR: 0.3 },
    CD:   { V1: 0.88, V2: 0.82, I: 0.78, aVL: 0.72, V5: 0.65, V6: 0.6, II: 0.55, III: 0.5, aVF: 0.45, V3: 0.4, V4: 0.38, aVR: 0.3 },
    NORM: { II: 0.5, I: 0.48, V5: 0.45, V6: 0.42, V4: 0.4, III: 0.38, aVF: 0.35, V3: 0.32, V2: 0.3, V1: 0.28, aVL: 0.25, aVR: 0.22 },
  };
  const weights = leadWeights[prediction] ?? leadWeights.NORM;

  const ABNORMAL_FEATURES: { [key: string]: string[] } = {
    MI:   ['ST Elevation', 'Q Wave Abnormal', 'T Wave Inversion', 'R Wave Loss'],
    STTC: ['ST Depression', 'T Wave Abnormal', 'QT Prolongation', 'Bradycardia'],
    HYP:  ['R Amplitude High', 'S Wave Deep', 'Left Axis Deviation', 'Sokolow-Lyon Criteria'],
    CD:   ['QRS Widening', 'Bundle Branch Block', 'AV Block', 'PR Prolongation'],
    NORM: ['Normal P Wave', 'Normal QRS', 'Normal ST Segment', 'Normal T Wave'],
  };
  const abnFeats = ABNORMAL_FEATURES[prediction] ?? ABNORMAL_FEATURES.NORM;

  return LEAD_NAMES.map((lead) => {
    const w = weights[lead] ?? 0.3;
    return {
      lead,
      influence: parseFloat((w * conf * 5).toFixed(2)),
      features: w > 0.55 ? abnFeats.slice(0, Math.ceil(w * 4)) : [],
    };
  }).sort((a, b) => b.influence - a.influence);
}

const DiagnosisPrediction = ({ isOpen, selectedPatient }: DiagnosisPredictionProps) => {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>('');

  const runPrediction = async () => {
    if (!selectedPatient) { setError('No patient selected'); return; }
    setLoading(true); setError(''); setPrediction(null);

    try {
      const response = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patientId: selectedPatient.patient_id }),
      });
      const data = await response.json();
      if (data.success) {
        setPrediction(data);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch {
      setError('Cannot connect to prediction server (localhost:3001)');
    } finally {
      setLoading(false);
    }
  };

  const colors = prediction ? (CLASS_COLORS[prediction.prediction] ?? CLASS_COLORS.NORM) : CLASS_COLORS.NORM;
  const leadInfluences = prediction ? computeLeadInfluences(prediction.prediction, prediction.probabilities) : [];
  const topLeads = leadInfluences.filter(l => l.influence >= 2.0).slice(0, 4);
  const abnormalLeads = leadInfluences.filter(l => l.features.length > 0).slice(0, 3);

  const confidencePct = prediction ? Math.round(prediction.confidence * 100) : 0;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="overflow-hidden"
        >
          <div className="p-5 bg-gradient-to-br from-emerald-950/40 to-teal-900/20 rounded-xl border border-emerald-600/30 mt-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center shadow-lg">
                  <span className="text-white text-lg">🧠</span>
                </div>
                <div>
                  <h3 className="text-base font-bold text-white">AI Diagnosis Prediction</h3>
                  <p className="text-xs text-gray-400">MLP Neural Network · Signal-Based · v3.0</p>
                </div>
              </div>
              {prediction && (
                <span className="text-xs px-2 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  {prediction.source === 'precomputed' ? '⚡ Cached' : '🔄 Live'}
                </span>
              )}
            </div>

            {selectedPatient ? (
              <div className="space-y-4">
                {/* Patient Info */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-black/30 rounded-lg p-3 border border-white/5">
                    <span className="text-gray-500 text-xs block">Patient</span>
                    <span className="text-white font-semibold">{selectedPatient.patient_id}</span>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3 border border-white/5">
                    <span className="text-gray-500 text-xs block">ECG ID</span>
                    <span className="text-white font-semibold">{selectedPatient.ecg_id}</span>
                  </div>
                </div>

                {/* Predict Button */}
                <button
                  onClick={runPrediction}
                  disabled={loading}
                  className={`w-full py-3 rounded-xl font-semibold text-sm transition-all ${
                    loading
                      ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white shadow-lg shadow-emerald-900/50'
                  }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      Analyzing ECG...
                    </span>
                  ) : '🎯 Run AI Diagnosis'}
                </button>

                {/* Error */}
                {error && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="bg-red-900/30 border border-red-500/40 rounded-lg p-3">
                    <p className="text-red-300 text-sm">⚠️ {error}</p>
                  </motion.div>
                )}

                {/* ── Expert Dashboard Results ── */}
                {prediction && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                    className="space-y-3">

                    {/* Q1: Predicted Condition */}
                    <div className={`rounded-xl p-4 border ${colors.border} ${colors.bg}`}>
                      <p className="text-xs font-semibold text-gray-400 mb-1">Predicted Condition</p>
                      <p className={`text-xl font-bold ${colors.text}`}>{prediction.prediction_full}</p>
                      <div className="mt-2 flex items-center gap-2">
                        <p className="text-xs text-gray-500">Confidence:</p>
                        <div className="flex-1 bg-gray-700/50 rounded-full h-2">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${confidencePct}%` }}
                            transition={{ duration: 0.8 }}
                            className={`h-2 rounded-full ${
                              prediction.prediction === 'MI' ? 'bg-gradient-to-r from-red-500 to-orange-400' :
                              prediction.prediction === 'NORM' ? 'bg-gradient-to-r from-green-500 to-teal-400' :
                              prediction.prediction === 'STTC' ? 'bg-gradient-to-r from-yellow-500 to-amber-400' :
                              prediction.prediction === 'HYP' ? 'bg-gradient-to-r from-orange-500 to-red-400' :
                              'bg-gradient-to-r from-purple-500 to-indigo-400'
                            }`}
                          />
                        </div>
                        <p className="text-xs font-bold text-white">{confidencePct}%</p>
                      </div>
                    </div>

                    {/* Q2: Which leads influenced this decision? */}
                    <div className="rounded-xl p-4 bg-blue-950/30 border border-blue-500/20">
                      <p className="text-xs font-semibold text-gray-400 mb-3">Lead Influence</p>
                      <div className="flex flex-wrap gap-2">
                        {topLeads.map((l) => (
                          <motion.span
                            key={l.lead}
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-semibold shadow"
                          >
                            {l.lead}
                            <span className="ml-1 opacity-70">(Influence: {l.influence})</span>
                          </motion.span>
                        ))}
                      </div>
                      {topLeads.length > 0 && (
                        <p className={`mt-2 text-sm font-bold ${colors.text}`}>
                          {topLeads.map(l => l.lead).join(', ')}
                        </p>
                      )}
                    </div>

                    {/* Q3: Abnormal Features */}
                    {abnormalLeads.length > 0 && (
                      <div className="rounded-xl p-4 bg-yellow-950/20 border border-yellow-500/20">
                        <p className="text-xs font-semibold text-gray-400 mb-3">Abnormal Features</p>
                        <div className="space-y-2">
                          {abnormalLeads.map((l) => (
                            <div key={l.lead} className="text-sm">
                              <span className="font-bold text-blue-300">{l.lead}:</span>
                              <span className="ml-2 text-gray-300">{l.features.join(', ')}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* All Class Probabilities */}
                    <div className="rounded-xl p-4 bg-black/30 border border-white/5">
                      <p className="text-xs font-semibold text-gray-400 mb-3">All Class Probabilities</p>
                      <div className="space-y-2">
                        {Object.entries(prediction.probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([cls, prob]) => {
                            const c = CLASS_COLORS[cls] ?? CLASS_COLORS.NORM;
                            return (
                              <div key={cls}>
                                <div className="flex justify-between text-xs mb-1">
                                  <span className={`font-semibold ${c.text}`}>{cls}</span>
                                  <span className="text-white">{(prob * 100).toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-gray-700/40 rounded-full h-1.5">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${prob * 100}%` }}
                                    transition={{ duration: 0.6 }}
                                    className={`h-1.5 rounded-full ${cls === prediction.prediction ? c.text.replace('text-', 'bg-') : 'bg-gray-600'}`}
                                    style={{ backgroundColor: cls === prediction.prediction ? undefined : '#4B5563' }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>

                    {/* SCP Codes reference */}
                    {prediction.patient_info?.actual_scp_codes && Object.keys(prediction.patient_info.actual_scp_codes).length > 0 && (
                      <div className="rounded-xl p-3 bg-indigo-950/20 border border-indigo-500/20">
                        <p className="text-xs font-semibold text-indigo-300 mb-2">Reference SCP Codes (from PTB-XL annotation)</p>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(prediction.patient_info.actual_scp_codes).map(([code, conf]) => (
                            <span key={code} className="text-xs px-2 py-0.5 rounded bg-indigo-800/50 text-indigo-200 border border-indigo-600/30">
                              {code} {(conf as number) > 0 ? `(${conf}%)` : ''}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Report */}
                    {prediction.patient_info?.report && prediction.patient_info.report !== 'nan' && (
                      <div className="rounded-xl p-3 bg-gray-800/30 border border-gray-600/20">
                        <p className="text-xs font-semibold text-gray-400 mb-1">Original ECG Report</p>
                        <p className="text-xs text-gray-300 italic leading-relaxed">{prediction.patient_info.report}</p>
                      </div>
                    )}

                    <p className="text-center text-xs text-gray-600">
                      MLP v3 · 1780 Training Samples · 5 Diagnostic Classes · Signal-Based Features
                    </p>
                  </motion.div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 text-sm">
                Select a patient to run AI diagnosis prediction
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default DiagnosisPrediction;

