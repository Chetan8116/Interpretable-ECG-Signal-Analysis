import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';
import { runECGProcessingPipeline, ProcessingResults } from '../utils/ecgSignalProcessing';

const LEAD_IDS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

const SignalProcessingViewer: React.FC = () => {
  const { selectedPatient, leads } = useECGStore();
  const [processingResults, setProcessingResults] = useState<ProcessingResults | null>(null);
  const [selectedLead, setSelectedLead] = useState<string>('II');
  const [loading, setLoading] = useState(false);

  const availableLeadIds = useMemo(
    () => leads.filter((lead) => lead.data.length > 0).map((lead) => lead.id),
    [leads],
  );

  useEffect(() => {
    if (!selectedPatient || availableLeadIds.length === 0) {
      return;
    }

    if (!availableLeadIds.includes(selectedLead)) {
      setSelectedLead(availableLeadIds[0]);
    }
  }, [availableLeadIds, selectedLead, selectedPatient]);

  useEffect(() => {
    const processSignal = () => {
      if (!selectedPatient || leads.length === 0) {
        setProcessingResults(null);
        return;
      }

      setLoading(true);
      try {
        const lead = leads.find((item) => item.id === selectedLead);
        if (!lead || lead.data.length === 0) {
          setProcessingResults(null);
          setLoading(false);
          return;
        }

        const results = runECGProcessingPipeline(lead.data, 500);
        setProcessingResults(results);
      } catch (error) {
        console.error('Error processing ECG signal:', error);
        setProcessingResults(null);
      } finally {
        setLoading(false);
      }
    };

    processSignal();
  }, [selectedPatient, leads, selectedLead]);

  if (!selectedPatient) {
    return (
      <div className="rounded-2xl bg-gradient-to-br from-white to-gray-50 p-8 shadow-xl">
        <p className="text-center text-gray-600">Select a patient to view signal processing</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="rounded-2xl bg-gradient-to-br from-white to-gray-50 p-8 shadow-xl">
        <div className="flex items-center justify-center space-x-3">
          <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-500" />
          <p className="text-gray-600">Processing ECG signal...</p>
        </div>
      </div>
    );
  }

  const renderMiniWaveform = (
    data: number[],
    color: string,
    label: string,
    overlayData?: number[],
    overlayLabel?: string,
    tagline?: string,
    overlayColor: string = 'black',
  ) => {
    if (!data || data.length === 0) return null;

    const displaySamples = 1000;
    const samples = data.slice(0, displaySamples);
    const validSamples = samples.filter((value) => Number.isFinite(value));

    if (validSamples.length === 0) {
      return (
        <div className="rounded-lg border-2 border-red-300 bg-white p-4">
          <h4 className="mb-2 text-sm font-bold text-red-600">{label}</h4>
          <div className="flex h-[100px] items-center justify-center rounded bg-red-50">
            <p className="text-xs text-red-600">Invalid data detected in this processing stage</p>
          </div>
        </div>
      );
    }

    const overlaySamples = overlayData ? overlayData.slice(0, displaySamples) : [];
    const validOverlaySamples = overlaySamples.filter((value) => Number.isFinite(value));
    const max = Math.max(...validSamples.map(Math.abs), ...validOverlaySamples.map(Math.abs), 1e-9);
    const scale = max > 0 ? 1 / max : 1;

    const points = samples
      .map((value, idx) => {
        const x = (idx / displaySamples) * 400;
        const safeValue = Number.isFinite(value) ? value : 0;
        const y = 50 - safeValue * scale * 40;
        return `${x},${y}`;
      })
      .join(' ');

    const overlayPoints =
      overlayData && overlayData.length > 0
        ? overlaySamples
            .map((value, idx) => {
              const x = (idx / displaySamples) * 400;
              const safeValue = Number.isFinite(value) ? value : 0;
              const y = 50 - safeValue * scale * 40;
              return `${x},${y}`;
            })
            .join(' ')
        : '';

    const minValue = Math.min(...validSamples);
    const maxValue = Math.max(...validSamples);
    const range = (maxValue - minValue).toFixed(3);

    return (
      <div className="rounded-lg border-2 border-gray-200 bg-white p-4 transition-colors hover:border-blue-300">
        <h4 className="mb-1 text-sm font-bold" style={{ color }}>
          {label}
        </h4>
        {tagline && <p className="mb-2 text-xs italic text-gray-500">{tagline}</p>}
        <svg width="100%" height="100" viewBox="0 0 400 100" preserveAspectRatio="none" className="bg-pink-50">
          {[0, 25, 50, 75, 100].map((y) => (
            <line key={`h${y}`} x1="0" y1={y} x2="400" y2={y} stroke="#ffb3b3" strokeWidth="0.5" />
          ))}
          {Array.from({ length: 17 }, (_, i) => i * 25).map((x) => (
            <line key={`v${x}`} x1={x} y1="0" x2={x} y2="100" stroke="#ffb3b3" strokeWidth="0.5" />
          ))}
          <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
          {overlayPoints && (
            <polyline points={overlayPoints} fill="none" stroke={overlayColor} strokeWidth="1.5" opacity="0.6" />
          )}
          <line x1="0" y1="50" x2="400" y2="50" stroke={color} strokeWidth="1" strokeDasharray="3,3" opacity="0.5" />
        </svg>
        <p className="mt-1 text-xs text-gray-600">
          {validSamples.length} samples | Range: {range} mV
          {overlayLabel && <span className="ml-2 text-black">• {overlayLabel}</span>}
        </p>
      </div>
    );
  };

  return (
    <div className="rounded-2xl bg-gradient-to-br from-white to-gray-50 p-8 shadow-xl">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-1 items-center space-x-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600">
            <svg className="h-7 w-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
              />
            </svg>
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-800">Signal Processing Pipeline</h2>
            <p className="text-sm text-gray-600">Medical-grade filtering at 500 Hz • PTB-XL Standard</p>
          </div>
        </div>

        <div className="flex items-center space-x-2 self-start lg:self-auto">
          <label className="text-sm font-semibold text-gray-700">Lead:</label>
          <select
            value={selectedLead}
            onChange={(e) => setSelectedLead(e.target.value)}
            className="rounded-lg border-2 border-gray-300 bg-white px-4 py-2 font-semibold text-gray-800 transition-colors hover:border-blue-400 focus:border-blue-500 focus:outline-none"
          >
            {LEAD_IDS.map((leadId) => {
              const hasData = availableLeadIds.includes(leadId);
              return (
                <option key={leadId} value={leadId} disabled={!hasData}>
                  {hasData ? `Lead ${leadId}` : `Lead ${leadId} (loading)`}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      <div className="mb-6 rounded-xl border-2 border-indigo-200 bg-gradient-to-r from-indigo-50 to-blue-50 p-4">
        <div className="flex items-start space-x-3">
          <svg className="mt-1 h-6 w-6 flex-shrink-0 text-indigo-600" fill="currentColor" viewBox="0 0 24 24">
            <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <h4 className="mb-1 font-bold text-indigo-900">7-Stage Professional ECG Processing</h4>
            <p className="text-sm text-indigo-800">
              This pipeline applies medical-grade filters used in clinical ECG analysis. Each stage removes
              specific artifacts while preserving critical cardiac waveform features, with a final overlap
              comparison against the original signal.
            </p>
          </div>
        </div>
      </div>

      {processingResults ? (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0 }}>
              {renderMiniWaveform(processingResults.raw, '#6b7280', '1. Raw Signal', undefined, undefined, 'All noise present')}
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              {renderMiniWaveform(processingResults.dcRemoved, '#ef4444', '2. DC Offset Removed', undefined, undefined, 'Centers signal around zero')}
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              {renderMiniWaveform(
                processingResults.baselineRemoved,
                '#f97316',
                '3. Baseline Wander Removed',
                processingResults.baselineEstimate,
                'baseline estimate',
                'Removes respiratory drift and movement artifacts',
              )}
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              {renderMiniWaveform(processingResults.notched, '#22c55e', '4. Notch Filter (50 Hz)', undefined, undefined, 'Power-line noise reduced')}
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
              {renderMiniWaveform(
                processingResults.bandpassed,
                '#3b82f6',
                '5. Bandpass (0.5-40 Hz)',
                undefined,
                undefined,
                'Muscle artifacts and sensor noise reduced',
              )}
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
              {renderMiniWaveform(processingResults.denoised, '#a855f7', '6. Wavelet Denoised', undefined, undefined, 'Cleaned signal output')}
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
              {renderMiniWaveform(
                processingResults.denoised,
                '#a855f7',
                '7. Raw vs Denoised Overlap',
                processingResults.raw,
                'gray = original, purple = denoised',
                'Feature preservation check against the original waveform',
                '#6b7280',
              )}
            </motion.div>
          </div>

          <div className="mt-6 rounded-xl border-2 border-emerald-200 bg-emerald-50 p-6">
            <h3 className="mb-4 text-xl font-bold text-emerald-900">Initial Feature Preservation Check</h3>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div>
                <p className="mb-1 text-xs text-emerald-700">Raw vs Denoised Correlation</p>
                <p className="font-bold text-emerald-900">{processingResults.featurePreservation.correlation.toFixed(3)}</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-emerald-700">Energy Retention</p>
                <p className="font-bold text-emerald-900">{processingResults.featurePreservation.energyRetentionPct.toFixed(1)}%</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-emerald-700">Peak Retention</p>
                <p className="font-bold text-emerald-900">{processingResults.featurePreservation.peakRetentionPct.toFixed(1)}%</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-emerald-700">Preservation Status</p>
                <p
                  className={`font-bold ${
                    processingResults.featurePreservation.isPreserved ? 'text-emerald-700' : 'text-amber-700'
                  }`}
                >
                  {processingResults.featurePreservation.isPreserved ? 'Preserved' : 'Review Needed'}
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-xl border-2 border-gray-300 bg-gray-50 p-6">
            <h3 className="mb-4 text-xl font-bold text-gray-800">Processing Parameters (500 Hz)</h3>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div>
                <p className="mb-1 text-xs text-gray-600">DC Removal</p>
                <p className="font-bold text-gray-800">Median</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Baseline Filter</p>
                <p className="font-bold text-gray-800">Cascade Median</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Spike Suppression</p>
                <p className="font-bold text-gray-800">25 ms</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Baseline Window</p>
                <p className="font-bold text-gray-800">200 ms</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Notch Frequency</p>
                <p className="font-bold text-gray-800">50 Hz</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Bandpass Range</p>
                <p className="font-bold text-gray-800">0.5 - 40 Hz</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Sample Rate</p>
                <p className="font-bold text-gray-800">500 Hz</p>
              </div>
              <div>
                <p className="mb-1 text-xs text-gray-600">Total Samples</p>
                <p className="font-bold text-gray-800">{processingResults.raw.length}</p>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 py-12 text-center">
          <svg className="mx-auto mb-3 h-16 w-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <p className="text-base font-semibold text-slate-700">Processing data is not ready for this lead yet.</p>
          <p className="mt-2 text-sm text-slate-500">
            Select another lead with loaded ECG data or wait a moment for patient signals to finish loading.
          </p>
        </div>
      )}
    </div>
  );
};

export default SignalProcessingViewer;
