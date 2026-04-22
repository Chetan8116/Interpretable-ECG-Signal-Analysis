import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';
import ECGWaveformWithHeatmap from './ECGWaveformWithHeatmap';

//  Types matching mlp_predictions.json 
interface TopFeature {
  col: string;
  label: string;
  attr: number;
}

interface AbnormalFeature {
  feature: string;
  lead: string;
  label: string;
  value: number;
  normalRange: string;
  status: 'Low' | 'High' | 'Normal';
  unit: string;
  attr: number;
}

interface TopLead {
  lead: string;
  influence: number;
  topFeatures: TopFeature[];
  abnormalFeatures: AbnormalFeature[];
}

interface MLPPrediction {
  ecg_id: number;
  trueClass?: string;          // ground-truth label (ResNet1D model only)
  predictedClass: string;
  predictedLabel: string;
  confidence: number;
  classProbabilities: Record<string, number>;
  topLeads: TopLead[];
  allAbnormal: AbnormalFeature[];
}

//  Condition colours 
const CONDITION_COLOR: Record<string, string> = {
  MI:   'text-red-600',
  CD:   'text-orange-600',
  HYP:  'text-yellow-700',
  STTC: 'text-blue-600',
  NORM: 'text-green-600',
};

const CONDITIONS: Record<string, { label: string; description: string }> = {
  MI:   { label: 'Myocardial Infarction',   description: 'Signs of myocardial ischaemia or infarction' },
  CD:   { label: 'Conduction Disturbance',  description: 'Bundle-branch block or AV conduction defect' },
  HYP:  { label: 'Hypertrophy',             description: 'Ventricular or atrial hypertrophy pattern' },
  STTC: { label: 'ST-T Change',             description: 'Non-specific ST-segment / T-wave abnormality' },
  NORM: { label: 'Normal Sinus Rhythm',     description: 'No significant ECG abnormality detected' },
};

//  Heatmap segment helpers 
// ── Feature → center offset from R-peak (samples @ 500 Hz) ──────────────────
// centerOffset: positive = after R, negative = before R
// barHalf: half-width of the vertical bar in samples (default ±8 = 16ms)
const BEAT_FEAT_MAP: { markerKey: string; defaultLabel: string; patterns: string[]; centerOffset: number; colour: string }[] = [
  { markerKey: 'R',  defaultLabel: 'R Peak',     patterns: ['R_amp', 'QRS', 'S_amp', 'Q_amp'], centerOffset: 0,    colour: 'rgba(220,38,38,{a})'  }, // at R peak
  { markerKey: 'ST', defaultLabel: 'ST Segment', patterns: ['ST_dev', 'ST_elev', 'ST_dep'],    centerOffset: 80,   colour: 'rgba(234,88,12,{a})'  }, // +160ms
  { markerKey: 'T',  defaultLabel: 'T Wave',     patterns: ['T_amp', 'T_dur', 'T_inv'],        centerOffset: 130,  colour: 'rgba(234,88,12,{a})'  }, // +260ms
  { markerKey: 'P',  defaultLabel: 'P Wave',     patterns: ['P_dur', 'P_amp', 'P_axis'],       centerOffset: -90,  colour: 'rgba(147,51,234,{a})' }, // -180ms
  { markerKey: 'PR', defaultLabel: 'PR Segment', patterns: ['PR_int', 'PR_seg'],               centerOffset: -50,  colour: 'rgba(59,130,246,{a})' }, // -100ms
  { markerKey: 'QT', defaultLabel: 'QT Interval',patterns: ['QT', 'QTc'],                       centerOffset: 100,  colour: 'rgba(59,130,246,{a})' }, // +200ms
  { markerKey: 'RR', defaultLabel: 'RR Interval',patterns: ['HR_bpm', 'RR_'],                   centerOffset: 0,    colour: 'rgba(16,185,129,{a})' }, // at R
];
const BAR_HALF = 8; // ±8 samples = 16 ms bar width
const MIN_HIGHLIGHT_LEADS = 2;
const MAX_HIGHLIGHT_LEADS = 4;
const GOOD_LEAD_THRESHOLD_RATIO = 0.72;

function getDynamicHighlightedLeadIds(topLeads: TopLead[]): string[] {
  if (!topLeads.length) return [];
  const maxInfluence = Math.max(topLeads[0].influence, 1e-6);

  const byThreshold = topLeads
    .filter(lead => lead.influence / maxInfluence >= GOOD_LEAD_THRESHOLD_RATIO)
    .slice(0, MAX_HIGHLIGHT_LEADS)
    .map(lead => lead.lead);

  if (byThreshold.length >= MIN_HIGHLIGHT_LEADS) {
    return byThreshold;
  }

  return topLeads.slice(0, MIN_HIGHLIGHT_LEADS).map(lead => lead.lead);
}

/** Simple Pan-Tompkins-inspired R-peak detector (pure JS, no deps).
 *  Returns sample indices of detected R-peaks. */
function detectRPeaks(data: number[], fs = 500): number[] {
  const n = data.length;
  if (n < fs) return [];

  // 1. Compute mean (DC) and subtract it
  const mean = data.reduce((s, v) => s + v, 0) / n;
  const centered = data.map(v => v - mean);

  // 2. Differentiate
  const diff = new Array(n).fill(0);
  for (let i = 2; i < n - 2; i++) {
    diff[i] = (-centered[i - 2] - 2 * centered[i - 1] + 2 * centered[i + 1] + centered[i + 2]) / 8;
  }

  // 3. Square
  const sq = diff.map(v => v * v);

  // 4. Moving-window integration (150 ms window)
  const win = Math.round(0.15 * fs);
  const mwi = new Array(n).fill(0);
  let windowSum = 0;
  for (let i = 0; i < n; i++) {
    windowSum += sq[i];
    if (i >= win) windowSum -= sq[i - win];
    mwi[i] = windowSum / win;
  }

  // 5. Threshold = 35% of max
  const maxMwi = Math.max(...mwi);
  const threshold = maxMwi * 0.35;

  // 6. Find peaks with refractory period of 200 ms
  const refractory = Math.round(0.20 * fs);
  const peaks: number[] = [];
  let lastPeak = -refractory;
  for (let i = 1; i < n - 1; i++) {
    if (mwi[i] > threshold && mwi[i] >= mwi[i - 1] && mwi[i] >= mwi[i + 1] && i - lastPeak > refractory) {
      // Refine: find actual maximum in original data within ±20 samples
      const lo = Math.max(0, i - 20);
      const hi = Math.min(n - 1, i + 20);
      let bestIdx = lo;
      for (let j = lo + 1; j <= hi; j++) {
        if (Math.abs(data[j]) > Math.abs(data[bestIdx])) bestIdx = j;
      }
      peaks.push(bestIdx);
      lastPeak = bestIdx;
    }
  }
  return peaks;
}

function buildHeatmapSegments(
  leadAbnormal: AbnormalFeature[],
  topFeatures: TopFeature[],
  leadData: number[],
  fs = 500
) {
  const totalSamples = leadData.length;
  if (!totalSamples) return [];

  // Detect actual R-peaks for beat-accurate positioning
  const rPeaks = detectRPeaks(leadData, fs);
  const beats: number[] = rPeaks.length >= 2 ? rPeaks : (() => {
    const fallbackRR = Math.round(fs * 0.85);
    const result: number[] = [];
    for (let r = fallbackRR; r < totalSamples - fallbackRR; r += fallbackRR) result.push(r);
    return result;
  })();

  // Collect influential feature candidates.
  type Candidate = { featureKey: string; label: string; attr: number };
  let candidates: Candidate[] = [
    ...leadAbnormal.map(af => ({ featureKey: af.feature, label: af.label, attr: af.attr ?? 0 })),
    ...topFeatures.map(tf => ({ featureKey: tf.col, label: tf.label, attr: tf.attr ?? 0 })),
  ].filter((c, idx, arr) => arr.findIndex(x => x.featureKey === c.featureKey) === idx); // unique

  if (!candidates.length || !beats.length) return [];

  // Sort by attr and map into unique marker families.
  candidates.sort((a, b) => b.attr - a.attr);
  type Marker = {
    markerKey: string;
    label: string;
    centerOffset: number;
    color: string;
    intensity: number;
  };

  const markers: Marker[] = [];
  const markerByKey = new Map<string, Marker>();

  candidates.forEach(candidate => {
    const rule = BEAT_FEAT_MAP.find(r =>
      r.patterns.some(p => candidate.featureKey.toUpperCase().includes(p.toUpperCase()))
    );
    if (!rule || markerByKey.has(rule.markerKey)) return;

    const intensity = Math.min(1, Math.max(0.55, candidate.attr * 80 + 0.4));
    const marker: Marker = {
      markerKey: rule.markerKey,
      label: candidate.label || rule.defaultLabel,
      centerOffset: rule.centerOffset,
      color: rule.colour.replace('{a}', intensity.toFixed(2)),
      intensity,
    };
    markerByKey.set(rule.markerKey, marker);
    markers.push(marker);
  });

  // Keep high-value markers while allowing multi-annotated beats.
  let selectedMarkers = markers.slice(0, 4);
  if (!selectedMarkers.length) return [];

  const hasP = selectedMarkers.some(m => m.markerKey === 'P');
  const hasR = selectedMarkers.some(m => m.markerKey === 'R');

  // If P is present, force the companion R marker (and vice versa) to show in the same lead.
  if (hasP !== hasR) {
    const companionRule = BEAT_FEAT_MAP.find(r => r.markerKey === (hasP ? 'R' : 'P'));
    if (companionRule) {
      selectedMarkers = [
        ...selectedMarkers,
        {
          markerKey: companionRule.markerKey,
          label: companionRule.defaultLabel,
          centerOffset: companionRule.centerOffset,
          color: companionRule.colour.replace('{a}', '0.75'),
          intensity: 0.75,
        },
      ];
    }
  }

  const maxAnnotatedBeats = 8;
  const beatStep = Math.max(1, Math.ceil(beats.length / maxAnnotatedBeats));
  const sampledBeats = beats.filter((_, idx) => idx % beatStep === 0).slice(0, maxAnnotatedBeats);
  const midBeat = sampledBeats[Math.floor(sampledBeats.length / 2)] ?? sampledBeats[0];

  const cyclePreR = Math.round(0.28 * fs);
  const cyclePostR = Math.round(0.42 * fs);
  const cycleStart = Math.max(0, midBeat - cyclePreR);
  const cycleEnd = Math.min(totalSamples - 1, midBeat + cyclePostR);

  const segments: Array<{ start: number; end: number; color: string; intensity: number; label?: string }> = [
    {
      start: cycleStart,
      end: cycleEnd,
      color: 'rgba(16,185,129,0.16)',
      intensity: 0.16,
      label: undefined,
    },
  ];

  const emittedLabels = new Set<string>();
  sampledBeats.forEach(beat => {
    selectedMarkers.forEach(marker => {
      const barCenter = beat + marker.centerOffset;
      if (barCenter < 0 || barCenter >= totalSamples) return;
      const barStart = Math.max(0, barCenter - BAR_HALF);
      const barEnd = Math.min(totalSamples - 1, barCenter + BAR_HALF);
      const label = emittedLabels.has(marker.label) ? undefined : marker.label;
      emittedLabels.add(marker.label);
      segments.push({
        start: barStart,
        end: barEnd,
        color: marker.color,
        intensity: marker.intensity,
        label,
      });
    });
  });

  return segments;
}

//  Component 
const ExpertDashboard: React.FC = () => {
  const { selectedPatient, leads, setHighlightedLeads } = useECGStore();
  const [prediction, setPrediction]    = useState<MLPPrediction | null>(null);
  const [isExpanded, setIsExpanded]    = useState(true);
  const [loading, setLoading]          = useState(false);
  const [highlightedLeads, setLocalHL] = useState<string[]>([]);
  const [predictionsCache, setPredictionsCache] = useState<Record<string, MLPPrediction> | null>(null);

  // Load the full predictions JSON once
  useEffect(() => {
    fetch('/mlp_predictions.json')
      .then(r => r.json())
      .then(data => setPredictionsCache(data))
      .catch(e => console.error('Failed to load mlp_predictions.json:', e));
  }, []);

  // Look up current patient when cache or patient changes
  useEffect(() => {
    if (!selectedPatient || !predictionsCache) {
      setPrediction(null);
      return;
    }

    setLoading(true);

    // Patient ID mapping: "PTB00001"  ecg_id 1
    const raw = selectedPatient.patientId;
    const numericId = parseInt(raw.replace(/\D/g, ''), 10);
    const pd = predictionsCache[String(numericId)] ?? null;
    setPrediction(pd);

    if (pd) {
      const selectedLeadIds = getDynamicHighlightedLeadIds(pd.topLeads);
      setLocalHL(selectedLeadIds);
      setHighlightedLeads(selectedLeadIds);
    }

    setLoading(false);
  }, [selectedPatient, predictionsCache]);

  //  Guard states 
  if (!selectedPatient) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">Select a patient to view expert analysis</p>
      </div>
    );
  }

  if (loading || !predictionsCache) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
          <p className="text-gray-600">Running MLP model</p>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">No MLP prediction available for this patient</p>
      </div>
    );
  }

  const cond = CONDITIONS[prediction.predictedClass] ?? { label: prediction.predictedLabel, description: '' };
  const condColour = CONDITION_COLOR[prediction.predictedClass] ?? 'text-gray-800';
  const top3Leads  = prediction.topLeads.slice(0, 3);
  const top3Names  = top3Leads.map(l => l.lead);

  const classColours: Record<string, string> = {
    MI:   'bg-red-500',
    CD:   'bg-orange-500',
    HYP:  'bg-yellow-500',
    STTC: 'bg-blue-500',
    NORM: 'bg-green-500',
  };

  return (
    <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">

      {/*  Header  */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
            <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-800">Expert Dashboard</h2>
            <p className="text-sm text-gray-600">MLP model  ECG_Diag_pipeline  PTB-XL</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className="px-4 py-2 rounded-lg bg-blue-100 border border-blue-200">
            <p className="text-xs text-blue-700 font-medium">Patient {selectedPatient.patientId}</p>
            <p className="text-xs text-blue-600">{selectedPatient.age}y, {selectedPatient.sex}</p>
          </div>
          <motion.button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <motion.svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"
              animate={{ rotate: isExpanded ? 180 : 0 }} transition={{ duration: 0.3 }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </motion.svg>
          </motion.button>
        </div>
      </div>

      {/*  Collapsible body  */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >

            {/* Q1 — Predicted condition  */}
            
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
              <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-xl p-6 border-2 border-red-200">
                <h3 className="text-lg font-bold text-gray-700 mb-2">Predicted Condition</h3>
                <div className="flex items-start justify-between mb-1">
                  <p className={`text-3xl font-bold ${condColour}`}>{cond.label}</p>
                  {prediction.trueClass && (
                    <div className={`flex items-center space-x-1 px-2 py-1 rounded-lg text-xs font-bold ${
                      prediction.trueClass === prediction.predictedClass
                        ? 'bg-green-100 text-green-700 border border-green-300'
                        : 'bg-red-100 text-red-700 border border-red-300'
                    }`}>
                      <span>{prediction.trueClass === prediction.predictedClass ? '✓' : '✗'}</span>
                      <span>True: {prediction.trueClass}</span>
                    </div>
                  )}
                </div>
                <p className="text-sm text-gray-500 italic mb-4">{cond.description}</p>

                <div className="flex items-center space-x-3 mb-5">
                  <span className="text-sm text-gray-600 w-24 shrink-0">Confidence:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-orange-400 to-red-500 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-bold text-gray-700 w-14 text-right">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="space-y-1">
                  {Object.entries(prediction.classProbabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, prob]) => (
                      <div key={cls} className="flex items-center space-x-2">
                        <span className="text-xs font-mono text-gray-500 w-10">{cls}</span>
                        <div className="flex-1 bg-gray-100 rounded h-2">
                          <div
                            className={`${classColours[cls] ?? 'bg-gray-400'} h-2 rounded transition-all duration-500`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600 w-10 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>
            </motion.div>

            {/* Q2 — Lead influence  */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }} className="mb-6">
              <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-6 border-2 border-blue-200">
                <h3 className="text-lg font-bold text-gray-700 mb-4">
                  Lead Influence
                </h3>
                <p className="text-2xl font-bold text-blue-700 mb-4">{top3Names.join(', ')}</p>

                <div className="flex flex-wrap gap-3 mb-4">
                  {prediction.topLeads.slice(0, 6).map((lead, idx) => (
                    <motion.div
                      key={lead.lead}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.05 * idx }}
                      className={`px-4 py-2 rounded-lg font-bold ${
                        idx === 0 ? 'bg-blue-600 text-white' :
                        idx === 1 ? 'bg-blue-400 text-white' :
                        idx === 2 ? 'bg-blue-300 text-white' :
                                    'bg-gray-200 text-gray-700'
                      }`}
                    >
                      {lead.lead}
                      <span className="ml-1 text-xs opacity-70">
                        ({(lead.influence * 100).toFixed(1)})
                      </span>
                    </motion.div>
                  ))}
                </div>

                <div className="space-y-1 mt-3">
                  {prediction.topLeads.slice(0, 6).map((lead, idx) => {
                    const maxInf = prediction.topLeads[0].influence;
                    return (
                      <div key={lead.lead} className="flex items-center space-x-2">
                        <span className="text-xs font-mono text-gray-500 w-8">{lead.lead}</span>
                        <div className="flex-1 bg-gray-100 rounded h-2">
                          <div
                            className={idx < 3 ? 'bg-blue-500 h-2 rounded' : 'bg-gray-300 h-2 rounded'}
                            style={{ width: `${(lead.influence / maxInf) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 w-14 text-right">
                          {(lead.influence * 100).toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>

            {/* Q3 — Abnormal features  */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }} className="mb-6">
              <div className="bg-gradient-to-r from-yellow-50 to-amber-50 rounded-xl p-6 border-2 border-yellow-200">
                <h3 className="text-lg font-bold text-gray-700 mb-4">
                  Abnormal Features
                </h3>
                <div className="space-y-2">
                  {top3Leads.map(lead => (
                    lead.abnormalFeatures.length > 0 ? (
                      <div key={lead.lead} className="bg-white/60 rounded-lg p-3">
                        <span className="font-bold text-gray-800">{lead.lead}:</span>
                        <span className="ml-2 text-gray-700">
                          {lead.abnormalFeatures
                            .map(af => `${af.label} (${af.value.toFixed(2)} ${af.unit}, ${af.status})`)
                            .join('  ')}
                        </span>
                      </div>
                    ) : (
                      <div key={lead.lead} className="bg-white/40 rounded-lg p-3">
                        <span className="font-bold text-gray-600">{lead.lead}:</span>
                        <span className="ml-2 text-gray-500 italic">No notable abnormalities</span>
                      </div>
                    )
                  ))}
                </div>
              </div>
            </motion.div>

            {/* Q4 — ECG waveforms  */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }} className="mb-6">
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-200">
                <h3 className="text-lg font-bold text-gray-700 mb-4">ECG Waveforms — Highlighted Leads</h3>

                {highlightedLeads.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {leads
                      .filter(lead => highlightedLeads.includes(lead.id))
                      .map((lead, idx) => {
                        const predLead = prediction.topLeads.find(l => l.lead === lead.id);
                        const segments = predLead && lead.data.length > 0
                          ? buildHeatmapSegments(predLead.abnormalFeatures, predLead.topFeatures, lead.data, 500)
                          : [];
                        const rPeakCount = lead.data.length > 0 ? detectRPeaks(lead.data, 500).length : 0;
                        const highlightedFeatures = segments.length > 0
                          ? [...new Set(segments.map(s => s.label).filter(Boolean))]
                          : [];
                        return (
                          <motion.div
                            key={lead.id}
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.1 * idx }}
                            className="bg-white rounded-lg p-4 border-2 border-green-400 shadow-lg relative overflow-hidden"
                          >
                            <div className="absolute top-2 right-2 z-10">
                              <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full font-bold">
                                #{idx + 1} Influence
                              </span>
                            </div>
                            <div className="mb-2">
                              <h4 className="font-bold text-lg" style={{ color: lead.color }}>Lead {lead.name}</h4>
                              {predLead && (
                                <>
                                  <p className="text-xs text-gray-500">
                                    Influence score: {(predLead.influence * 100).toFixed(2)}
                                    {rPeakCount > 0 && <span className="ml-2 text-green-600 font-semibold">• {rPeakCount} beats detected</span>}
                                  </p>
                                  {highlightedFeatures.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-1">
                                      {highlightedFeatures.map((f, i) => (
                                        <span key={i} className="px-1.5 py-0.5 bg-orange-100 text-orange-700 text-[10px] rounded font-medium">{f}</span>
                                      ))}
                                    </div>
                                  )}
                                </>
                              )}
                            </div>
                            {lead.data.length > 0 ? (
                              <>
                                <div className="h-32">
                                  <ECGWaveformWithHeatmap
                                    data={lead.data}
                                    leadName={lead.id}
                                    color={lead.color}
                                    animate={false}
                                    speed={1}
                                    amplitude={1}
                                    width={350}
                                    height={120}
                                    heatmapSegments={segments}
                                  />
                                </div>
                              </>
                            ) : (
                              <div className="h-32 bg-gray-100 rounded flex items-center justify-center">
                                <div className="text-center">
                                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500 mx-auto mb-2" />
                                  <p className="text-xs text-gray-500">Loading ECG data</p>
                                </div>
                              </div>
                            )}
                          </motion.div>
                        );
                      })}
                  </div>
                ) : (
                  <div className="bg-white rounded-lg p-6 text-center">
                    <p className="text-gray-500 italic">No leads to highlight</p>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Q5 — Feature vs normal range table  */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}>
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border-2 border-purple-200">
                <h3 className="text-lg font-bold text-gray-700 mb-4">Values vs Normal Range</h3>

                {prediction.allAbnormal.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full bg-white rounded-lg overflow-hidden">
                      <thead className="bg-gradient-to-r from-purple-600 to-pink-600 text-white">
                        <tr>
                          <th className="px-4 py-3 text-left font-bold">Lead</th>
                          <th className="px-4 py-3 text-left font-bold">Feature</th>
                          <th className="px-4 py-3 text-right font-bold">Value</th>
                          <th className="px-4 py-3 text-right font-bold">Normal</th>
                          <th className="px-4 py-3 text-center font-bold">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {prediction.allAbnormal.slice(0, 12).map((af, idx) => (
                          <motion.tr
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.04 * idx }}
                            className={`border-b ${idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'} hover:bg-purple-50 transition-colors`}
                          >
                            <td className="px-4 py-3 font-bold text-gray-700">{af.lead}</td>
                            <td className="px-4 py-3 text-gray-800">{af.label}</td>
                            <td className="px-4 py-3 text-right font-mono font-bold text-gray-900">
                              {af.value.toFixed(af.unit === 'ms' ? 1 : 3)} {af.unit}
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-gray-500">
                              {af.normalRange} {af.unit}
                            </td>
                            <td className="px-4 py-3 text-center">
                              <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                                af.status === 'High' ? 'bg-red-200 text-red-800' :
                                af.status === 'Low'  ? 'bg-yellow-200 text-yellow-800' :
                                                       'bg-green-200 text-green-800'
                              }`}>
                                {af.status}
                              </span>
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="bg-white rounded-lg p-6 text-center">
                    <svg className="w-14 h-14 text-green-400 mx-auto mb-3" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                    </svg>
                    <p className="text-lg font-semibold text-green-600">All features within normal range</p>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Summary box  */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.55 }}
              className="mt-6 bg-gradient-to-r from-indigo-100 to-purple-100 rounded-xl p-6 border-2 border-indigo-300">
              <h4 className="font-bold text-gray-800 mb-3">Clinical Summary</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white/70 rounded-lg p-3">
                  <p className="text-xs text-gray-600">Diagnosis</p>
                  <p className={`text-lg font-bold ${condColour}`}>{prediction.predictedClass}</p>
                </div>
                <div className="bg-white/70 rounded-lg p-3">
                  <p className="text-xs text-gray-600">Confidence</p>
                  <p className="text-2xl font-bold text-blue-600">{(prediction.confidence * 100).toFixed(0)}%</p>
                </div>
                <div className="bg-white/70 rounded-lg p-3">
                  <p className="text-xs text-gray-600">Abnormal Features</p>
                  <p className="text-2xl font-bold text-orange-600">{prediction.allAbnormal.length}</p>
                </div>
                <div className="bg-white/70 rounded-lg p-3">
                  <p className="text-xs text-gray-600">Top Lead</p>
                  <p className="text-2xl font-bold text-purple-600">{prediction.topLeads[0]?.lead ?? '—'}</p>
                </div>
              </div>
            </motion.div>

            {/* Disclaimer  */}
            <div className="mt-6 bg-yellow-50 border border-yellow-300 rounded-lg p-4">
              <p className="text-sm text-yellow-800">
                <strong> Clinical Decision Support:</strong> Predictions come from an MLP trained on the
                PTB-XL dataset. All findings must be verified by qualified medical professionals and are
                not intended for direct clinical use.
              </p>
            </div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ExpertDashboard;
