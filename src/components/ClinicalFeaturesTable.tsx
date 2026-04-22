import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';

interface ClinicalFeatures {
  lead_name: string;
  P_duration: number;
  P_amplitude: number;
  PR_interval: number;
  QRS_width: number;
  R_amplitude: number;
  ST_elevation: number;
  T_amplitude: number;
  QT_interval: number;
  Heart_rate: number;
}

const ClinicalFeaturesTable: React.FC = () => {
  const { selectedPatient } = useECGStore();
  const [features, setFeatures] = useState<ClinicalFeatures[]>([]);
  const [loading, setLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const loadFeatures = async () => {
      if (!selectedPatient) {
        setLoading(false);
        return;
      }

      try {
        console.log('Loading features for patient:', selectedPatient.patientId);
        
        // Extract numeric ID from patientId (e.g., "PTB00001" -> 1)
        const numericId = parseInt(selectedPatient.patientId.replace('PTB', ''));
        console.log('Numeric ecg_id extracted:', numericId);
        
        const response = await fetch('/ptbxl_records.json');
        const records = await response.json();
        
        console.log('Total records loaded:', records.length);
        console.log('First few ecg IDs:', records.slice(0, 5).map((r: any) => r.ecg_id));
        
        // Find the patient record by ecg_id (NOT patient_id)
        const patientRecord = records.find((r: any) => r.ecg_id === numericId);
        
        console.log('Patient record found:', !!patientRecord);
        
        if (!patientRecord) {
          console.log('Patient not found in records');
          setFeatures([]);
          setLoading(false);
          return;
        }
        
        if (!patientRecord.leads) {
          console.log('Patient record has no leads');
          setFeatures([]);
          setLoading(false);
          return;
        }
        
        console.log('Patient record leads:', Object.keys(patientRecord.leads));
        
        // Extract clinical features for each lead
        const leadNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
        const extractedFeatures: ClinicalFeatures[] = [];
        
        leadNames.forEach(leadName => {
          const leadData = patientRecord.leads[leadName];
          if (leadData && leadData.clinical_features) {
            const cf = leadData.clinical_features;
            const beatRate = leadData.beat_detection?.beat_rate || 0;
            extractedFeatures.push({
              lead_name: leadName,
              P_duration: cf.P_duration_mean || 0,
              P_amplitude: cf.P_amplitude_mean || 0,
              PR_interval: cf.PR_interval_mean || 0,
              QRS_width: cf.QRS_width_mean || 0,
              R_amplitude: cf.R_amplitude_mean || 0,
              ST_elevation: cf.ST_elevation_mean || 0,
              T_amplitude: cf.T_amplitude_mean || 0,
              QT_interval: cf.QT_interval_mean || 0,
              Heart_rate: beatRate,
            });
          }
        });
        
        console.log('Extracted features for', extractedFeatures.length, 'leads');
        setFeatures(extractedFeatures);
      } catch (error) {
        console.error('Error loading clinical features:', error);
        setFeatures([]);
      } finally {
        setLoading(false);
      }
    };

    loadFeatures();
  }, [selectedPatient]);

  if (!selectedPatient) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">Select a patient to view clinical features</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="text-gray-600">Loading clinical features...</p>
        </div>
      </div>
    );
  }

  if (features.length === 0) {
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
        <p className="text-gray-600 text-center">No clinical features available for this patient</p>
      </div>
    );
  }

  // Define feature columns to display
  const featureColumns = [
    { key: 'P_duration', label: 'P_dur', unit: 'ms', description: 'P Wave Duration', scale: 1000 },
    { key: 'QRS_width', label: 'QRS_w', unit: 'ms', description: 'QRS Width', scale: 1000 },
    { key: 'QT_interval', label: 'QT', unit: 'ms', description: 'QT Interval', scale: 1000 },
    { key: 'ST_elevation', label: 'ST', unit: 'mV', description: 'ST Elevation', scale: 1 },
    { key: 'R_amplitude', label: 'R_amp', unit: 'mV', description: 'R Wave Amplitude', scale: 1 },
    { key: 'T_amplitude', label: 'T_amp', unit: 'mV', description: 'T Wave Amplitude', scale: 1 },
    { key: 'PR_interval', label: 'PR', unit: 'ms', description: 'PR Interval', scale: 1000 },
    { key: 'Heart_rate', label: 'HR', unit: 'bpm', description: 'Heart Rate', scale: 1 },
  ];

  return (
    <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Clinical Features Table</h2>
            <p className="text-sm text-gray-500">Patient {selectedPatient.patientId} • {features.length} Leads</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className="px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full text-sm font-semibold">
            Output = Feature table
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
            {/* Table */}
            <div className="overflow-x-auto">
        <motion.table
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="w-full"
        >
          <thead>
            <tr className="border-b-2 border-gray-200">
              <th className="px-4 py-3 text-left font-semibold text-gray-700 bg-gradient-to-r from-blue-50 to-blue-100 rounded-tl-xl">
                Lead
              </th>
              {featureColumns.map((col, idx) => (
                <th
                  key={col.key}
                  className={`px-4 py-3 text-center font-semibold text-gray-700 bg-gradient-to-r from-blue-50 to-blue-100 ${
                    idx === featureColumns.length - 1 ? 'rounded-tr-xl' : ''
                  }`}
                  title={col.description}
                >
                  <div className="flex flex-col items-center">
                    <span>{col.label}</span>
                    <span className="text-xs text-gray-500 font-normal">({col.unit})</span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feature, rowIdx) => (
              <motion.tr
                key={`${feature.lead_name}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: rowIdx * 0.05 }}
                className={`border-b border-gray-200 hover:bg-blue-50/50 transition-colors ${
                  rowIdx === features.length - 1 ? 'border-b-0' : ''
                }`}
              >
                <td className="px-4 py-3 font-semibold text-gray-800 bg-gradient-to-r from-emerald-50 to-emerald-100">
                  {feature.lead_name}
                </td>
                {featureColumns.map((col) => {
                  const value = feature[col.key as keyof ClinicalFeatures];
                  const numValue = typeof value === 'number' ? value * col.scale : 0;
                  
                  // Color coding based on typical clinical ranges
                  let colorClass = 'text-gray-800';
                  
                  if (col.key === 'ST_elevation') {
                    if (Math.abs(numValue) > 0.1) colorClass = 'text-red-600 font-semibold';
                  } else if (col.key === 'QT_interval') {
                    if (numValue > 440) colorClass = 'text-orange-600 font-semibold';
                  } else if (col.key === 'Heart_rate') {
                    if (numValue < 60 || numValue > 100) colorClass = 'text-orange-600 font-semibold';
                  }
                  
                  return (
                    <td
                      key={col.key}
                      className={`px-4 py-3 text-center ${colorClass}`}
                    >
                      {numValue.toFixed(2)}
                    </td>
                  );
                })}
              </motion.tr>
            ))}
          </tbody>
        </motion.table>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
            <span className="text-gray-600">Normal range</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-orange-500"></div>
            <span className="text-gray-600">Borderline</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span className="text-gray-600">Abnormal</span>
          </div>
          <div className="text-gray-500 text-right md:col-start-4">
            12-lead analysis
          </div>
        </div>
      </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ClinicalFeaturesTable;
