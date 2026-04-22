import React, { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';

interface FeatureLocation {
  feature: string;
  lead: string;
  startSample: number;
  endSample: number;
  shapValue: number;
  featureType: 'P' | 'QRS' | 'ST' | 'T' | 'QT';
}

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

const FEATURE_STYLES: Record<FeatureLocation['featureType'], { label: string; bar: string; card: string }> = {
  P: { label: 'P Wave', bar: 'bg-purple-500', card: 'border-purple-300 bg-gradient-to-br from-purple-100 to-purple-50' },
  QRS: { label: 'QRS Complex', bar: 'bg-red-500', card: 'border-red-300 bg-gradient-to-br from-red-100 to-red-50' },
  ST: { label: 'ST Segment', bar: 'bg-orange-500', card: 'border-orange-300 bg-gradient-to-br from-orange-100 to-orange-50' },
  T: { label: 'T Wave', bar: 'bg-blue-500', card: 'border-blue-300 bg-gradient-to-br from-blue-100 to-blue-50' },
  QT: { label: 'QT Interval', bar: 'bg-violet-500', card: 'border-violet-300 bg-gradient-to-br from-violet-100 to-violet-50' },
};

const ECGHeatmapViewer: React.FC = () => {
  const { selectedPatient } = useECGStore();
  const [featureLocations, setFeatureLocations] = useState<FeatureLocation[]>([]);
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const buildFeatureHeatmap = () => {
      if (!selectedPatient) {
        setFeatureLocations([]);
        return;
      }

      setLoading(true);
      try {
        const samplingRate = 500;
        const beatStart = 100;
        const locations: FeatureLocation[] = [];

        LEAD_NAMES.forEach((leadName, index) => {
          const beats = selectedPatient.leads[index];
          const features = beats?.[0]?.features;
          if (!features) {
            return;
          }

          const pDuration = Math.max(0.04, features.P_duration || 0);
          const qrsDuration = Math.max(0.05, features.QRS_duration || 0);
          const stDuration = Math.max(0.08, features.ST_duration || 0.12);
          const qtDuration = Math.max(0.24, features.QT_interval || 0.4);

          const featureCandidates: FeatureLocation[] = [
            {
              feature: `${leadName}_P_duration`,
              lead: leadName,
              startSample: beatStart,
              endSample: beatStart + Math.floor(pDuration * samplingRate),
              shapValue: Math.abs((features.P_duration || 0.08) - 0.09) * 4 + Math.abs(features.P_amp || 0) * 0.6,
              featureType: 'P',
            },
            {
              feature: `${leadName}_QRS_width`,
              lead: leadName,
              startSample: beatStart + Math.floor(0.16 * samplingRate),
              endSample: beatStart + Math.floor((0.16 + qrsDuration) * samplingRate),
              shapValue: Math.abs((features.QRS_duration || 0.08) - 0.09) * 6 + Math.abs(features.R_amp || 0) * 0.35,
              featureType: 'QRS',
            },
            {
              feature: `${leadName}_ST_segment`,
              lead: leadName,
              startSample: beatStart + Math.floor(0.24 * samplingRate),
              endSample: beatStart + Math.floor((0.24 + stDuration) * samplingRate),
              shapValue: Math.abs(features.ST_slope || 0) * 28,
              featureType: 'ST',
            },
            {
              feature: `${leadName}_T_amplitude`,
              lead: leadName,
              startSample: beatStart + Math.floor(0.36 * samplingRate),
              endSample: beatStart + Math.floor(0.52 * samplingRate),
              shapValue: Math.abs((features.T_amp || 0) - 0.3) * 1.2,
              featureType: 'T',
            },
            {
              feature: `${leadName}_QT_interval`,
              lead: leadName,
              startSample: beatStart + Math.floor(0.16 * samplingRate),
              endSample: beatStart + Math.floor((0.16 + qtDuration) * samplingRate),
              shapValue: Math.abs((features.QT_interval || 0.4) - 0.4) * 3,
              featureType: 'QT',
            },
          ];

          featureCandidates
            .filter((item) => item.shapValue > 0.015 && item.endSample > item.startSample)
            .forEach((item) => locations.push(item));
        });

        locations.sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
        setFeatureLocations(locations.slice(0, 15));
      } catch (error) {
        console.error('Error building feature heatmap:', error);
        setFeatureLocations([]);
      } finally {
        setLoading(false);
      }
    };

    buildFeatureHeatmap();
  }, [selectedPatient]);

  const maxShap = useMemo(
    () => Math.max(...featureLocations.map((item) => Math.abs(item.shapValue)), 0.001),
    [featureLocations],
  );

  if (!selectedPatient) {
    return (
      <div className="rounded-2xl bg-gradient-to-br from-white to-gray-50 p-8 shadow-xl">
        <p className="text-center text-gray-600">Select a patient to view ECG heatmap</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="rounded-2xl bg-gradient-to-br from-white to-gray-50 p-8 shadow-xl">
        <div className="flex items-center justify-center space-x-3">
          <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-purple-500" />
          <p className="text-gray-600">Building ECG heatmap...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-gradient-to-br from-white to-gray-50 p-8 shadow-xl">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-orange-500 to-red-600">
            <svg className="h-7 w-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-800">ECG Segment Heatmap</h2>
            <p className="text-sm text-gray-600">Feature-to-waveform localization derived from loaded patient features</p>
          </div>
        </div>

        <motion.button
          onClick={() => setIsExpanded(!isExpanded)}
          className="rounded-lg bg-gray-100 p-2 transition-colors hover:bg-gray-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <motion.svg
            className="h-6 w-6 text-gray-700"
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

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="mb-6 rounded-xl border-2 border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 p-4">
              <div className="flex items-start space-x-3">
                <svg className="mt-1 h-6 w-6 flex-shrink-0 text-blue-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <h4 className="mb-1 font-bold text-blue-900">How This Works</h4>
                  <p className="text-sm text-blue-800">
                    Each colored segment marks the waveform region most strongly associated with important ECG features
                    already loaded for the selected patient. Higher intensity reflects stronger model-facing signal change.
                  </p>
                </div>
              </div>
            </div>

            {featureLocations.length > 0 ? (
              <>
                <div className="mb-6 overflow-x-auto">
                  <table className="w-full overflow-hidden rounded-lg bg-white shadow">
                    <thead className="bg-gradient-to-r from-orange-600 to-red-600 text-white">
                      <tr>
                        <th className="px-4 py-3 text-left font-bold">#</th>
                        <th className="px-4 py-3 text-left font-bold">Lead</th>
                        <th className="px-4 py-3 text-left font-bold">Feature Type</th>
                        <th className="px-4 py-3 text-center font-bold">Sample Range</th>
                        <th className="px-4 py-3 text-center font-bold">Duration</th>
                        <th className="px-4 py-3 text-right font-bold">Score</th>
                        <th className="px-4 py-3 text-center font-bold">Importance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {featureLocations.slice(0, 10).map((loc, idx) => {
                        const duration = (((loc.endSample - loc.startSample) / 500) * 1000).toFixed(0);
                        const importance = (Math.abs(loc.shapValue) / maxShap) * 100;
                        return (
                          <motion.tr
                            key={loc.feature}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            className={`${idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'} border-b transition-colors hover:bg-orange-50`}
                          >
                            <td className="px-4 py-3 font-bold text-gray-700">{idx + 1}</td>
                            <td className="px-4 py-3">
                              <span className="rounded-full bg-blue-100 px-3 py-1 text-sm font-bold text-blue-800">{loc.lead}</span>
                            </td>
                            <td className="px-4 py-3">
                              <span className={`rounded-full px-3 py-1 text-sm font-semibold ${FEATURE_STYLES[loc.featureType].card}`}>
                                {FEATURE_STYLES[loc.featureType].label}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-center font-mono text-sm text-gray-700">
                              [{loc.startSample} - {loc.endSample}]
                            </td>
                            <td className="px-4 py-3 text-center text-gray-700">{duration} ms</td>
                            <td className="px-4 py-3 text-right font-bold text-gray-900">{loc.shapValue.toFixed(3)}</td>
                            <td className="px-4 py-3">
                              <div className="flex items-center justify-center space-x-2">
                                <div className="h-3 w-24 rounded-full bg-gray-200">
                                  <div
                                    className="h-3 rounded-full bg-gradient-to-r from-orange-400 to-red-500 transition-all"
                                    style={{ width: `${importance}%` }}
                                  />
                                </div>
                                <span className="text-sm font-bold text-gray-700">{importance.toFixed(0)}%</span>
                              </div>
                            </td>
                          </motion.tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="mb-6">
                  <h3 className="mb-4 text-xl font-bold text-gray-800">Color Legend</h3>
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
                    {(['P', 'QRS', 'ST', 'T', 'QT'] as FeatureLocation['featureType'][]).map((type, idx) => (
                      <motion.div
                        key={type}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: idx * 0.08 }}
                        className={`rounded-lg border-2 p-4 ${FEATURE_STYLES[type].card}`}
                      >
                        <div className={`mb-2 h-3 w-full rounded-full ${FEATURE_STYLES[type].bar} opacity-75`} />
                        <h4 className="text-sm font-bold text-gray-800">{FEATURE_STYLES[type].label}</h4>
                        <p className="mt-1 text-xs text-gray-600">
                          {type === 'P' && 'Atrial depolarization'}
                          {type === 'QRS' && 'Ventricular depolarization'}
                          {type === 'ST' && 'Ischemic change zone'}
                          {type === 'T' && 'Ventricular repolarization'}
                          {type === 'QT' && 'Global timing interval'}
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                  <div className="rounded-lg border-2 border-red-300 bg-gradient-to-br from-red-100 to-red-50 p-4">
                    <p className="mb-1 text-sm text-red-700">QRS Segments</p>
                    <p className="text-3xl font-bold text-red-800">{featureLocations.filter((f) => f.featureType === 'QRS').length}</p>
                  </div>
                  <div className="rounded-lg border-2 border-orange-300 bg-gradient-to-br from-orange-100 to-orange-50 p-4">
                    <p className="mb-1 text-sm text-orange-700">ST Segments</p>
                    <p className="text-3xl font-bold text-orange-800">{featureLocations.filter((f) => f.featureType === 'ST').length}</p>
                  </div>
                  <div className="rounded-lg border-2 border-purple-300 bg-gradient-to-br from-purple-100 to-purple-50 p-4">
                    <p className="mb-1 text-sm text-purple-700">P Waves</p>
                    <p className="text-3xl font-bold text-purple-800">{featureLocations.filter((f) => f.featureType === 'P').length}</p>
                  </div>
                  <div className="rounded-lg border-2 border-blue-300 bg-gradient-to-br from-blue-100 to-blue-50 p-4">
                    <p className="mb-1 text-sm text-blue-700">Total Features</p>
                    <p className="text-3xl font-bold text-blue-800">{featureLocations.length}</p>
                  </div>
                </div>
              </>
            ) : (
              <div className="py-8 text-center">
                <svg className="mx-auto mb-3 h-16 w-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-gray-600">No significant feature segments detected for the currently loaded patient.</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ECGHeatmapViewer;
