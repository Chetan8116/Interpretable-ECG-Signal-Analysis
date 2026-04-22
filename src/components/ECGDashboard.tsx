import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';
import ECGWaveform from './ECGWaveform';
import ECGGenerator from '../utils/ecgGenerator';
import DenoisingPanel from './DenoisingPanel';

// Inline waveform generation from features with lead-specific characteristics
function generateECGFromFeatures(features: any, samplingRate: number = 500, leadId: string = 'II'): number[] {
  const duration = 1.0; // 1 second
  const samples = Math.floor(duration * samplingRate);
  const ecgData: number[] = [];

  // Lead-specific modifiers based on medical ECG characteristics
  const leadModifiers: { [key: string]: { pMult: number, rMult: number, tMult: number, invert: boolean } } = {
    'I': { pMult: 1.0, rMult: 1.0, tMult: 1.0, invert: false },
    'II': { pMult: 1.2, rMult: 1.5, tMult: 1.2, invert: false },      // Usually largest amplitude
    'III': { pMult: 0.6, rMult: 0.8, tMult: 0.7, invert: false },
    'aVR': { pMult: -0.8, rMult: -1.2, tMult: -0.9, invert: true },  // Typically inverted
    'aVL': { pMult: 0.9, rMult: 1.1, tMult: 0.8, invert: false },
    'aVF': { pMult: 1.0, rMult: 1.3, tMult: 1.0, invert: false },
    'V1': { pMult: 0.5, rMult: -0.6, tMult: 0.9, invert: false },    // Small R, deep S
    'V2': { pMult: 0.6, rMult: 0.4, tMult: 1.0, invert: false },     // Transitional
    'V3': { pMult: 0.7, rMult: 1.0, tMult: 1.1, invert: false },     // Balanced
    'V4': { pMult: 0.8, rMult: 1.4, tMult: 1.2, invert: false },     // Tall R
    'V5': { pMult: 0.9, rMult: 1.6, tMult: 1.1, invert: false },     // Large amplitude
    'V6': { pMult: 0.8, rMult: 1.5, tMult: 1.0, invert: false }
  };

  const modifier = leadModifiers[leadId] || { pMult: 1.0, rMult: 1.0, tMult: 1.0, invert: false };

  // Normalize durations to prevent invalid values
  const P_duration = Math.max(0.04, Math.min(0.12, features.P_duration || 0.08));
  const QRS_duration = Math.max(0.06, Math.min(0.12, features.QRS_duration || 0.08));
  const ST_duration = Math.max(0.08, Math.min(0.20, features.ST_duration || 0.12));
  const QT_interval = Math.max(0.30, Math.min(0.50, features.QT_interval || 0.40));

  for (let i = 0; i < samples; i++) {
    const t = i / samplingRate;
    let signal = 0;

    // P wave (atrial depolarization) - with lead-specific amplitude
    if (t >= 0.0 && t < P_duration) {
      const pPhase = t / P_duration;
      signal += features.P_amp * modifier.pMult * Math.sin(pPhase * Math.PI);
    }

    // PR segment (isoelectric)
    const prEnd = P_duration + 0.05;

    // QRS complex (ventricular depolarization) - with lead-specific characteristics
    const qrsStart = prEnd;
    const qrsEnd = qrsStart + QRS_duration;
    if (t >= qrsStart && t < qrsEnd) {
      const qrsPhase = (t - qrsStart) / QRS_duration;
      
      // Q wave (negative deflection)
      if (qrsPhase < 0.2) {
        signal -= Math.abs(features.R_amp) * 0.25 * modifier.rMult * Math.sin(qrsPhase * Math.PI / 0.2);
      }
      // R wave (main deflection - lead-specific amplitude and polarity)
      else if (qrsPhase < 0.5) {
        const rPhase = (qrsPhase - 0.2) / 0.3;
        signal += features.R_amp * modifier.rMult * Math.sin(rPhase * Math.PI);
      }
      // S wave (negative deflection after R - more prominent in some leads)
      else {
        const sPhase = (qrsPhase - 0.5) / 0.5;
        const sMult = (leadId === 'V1' || leadId === 'V2') ? 1.5 : 0.4; // Deeper S in V1/V2
        signal -= Math.abs(features.R_amp) * sMult * modifier.rMult * Math.sin(sPhase * Math.PI);
      }
    }

    // ST segment (early repolarization)
    const stStart = qrsEnd;
    const stEnd = stStart + ST_duration;
    if (t >= stStart && t < stEnd) {
      const stPhase = (t - stStart) / ST_duration;
      // ST slope represents elevation/depression
      signal += features.ST_slope * stPhase * 10; // Amplify ST changes
    }

    // T wave (ventricular repolarization) - with lead-specific amplitude
    const tStart = stEnd;
    const tDuration = 0.16;
    const tEnd = tStart + tDuration;
    if (t >= tStart && t < tEnd) {
      const tPhase = (t - tStart) / tDuration;
      signal += features.T_amp * modifier.tMult * Math.sin(tPhase * Math.PI);
    }

    // Apply inversion for aVR and potentially V1
    if (modifier.invert) {
      signal = -signal;
    }

    ecgData.push(signal);
  }

  return ecgData;
}

const ECGDashboard: React.FC = () => {
  const { 
    leads, 
    isRecording, 
    isAnimating,
    speed, 
    amplitude, 
    selectedPatient, 
    useRealData,
    highlightedLeads,
    updateLeadData, 
    setHeartRate,
    setLeads,
    setAnimating,
    setRecording,
    setSpeed,
    setAmplitude
  } = useECGStore();
  const [generator] = useState(() => new ECGGenerator({ heartRate: 75, amplitude }));
  const [isInitialized, setIsInitialized] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [showSettings, setShowSettings] = useState(false);

  // Reset initialization when patient changes
  useEffect(() => {
    setIsInitialized(false);
    setRecordingTime(0);
  }, [selectedPatient]);

  // Track recording time
  useEffect(() => {
    if (!isRecording) {
      setRecordingTime(0);
      return;
    }

    const startTime = Date.now();
    const interval = setInterval(() => {
      setRecordingTime((Date.now() - startTime) / 1000);
    }, 100);

    return () => clearInterval(interval);
  }, [isRecording]);

  // Load patient data when selected
  useEffect(() => {
    if (useRealData && selectedPatient && !isInitialized) {
      console.log('Loading patient data:', selectedPatient.patientId);
      
      const leadMapping: { [key: string]: number } = {
        'I': 0, 'II': 1, 'III': 2,
        'aVR': 3, 'aVL': 4, 'aVF': 5,
        'V1': 6, 'V2': 7, 'V3': 8,
        'V4': 9, 'V5': 10, 'V6': 11
      };

      // Use REAL signal data from patient instead of generating synthetic waveforms
      const updatedLeads = leads.map(lead => {
        const leadIdx = leadMapping[lead.id];
        
        if (selectedPatient.leads[leadIdx] && selectedPatient.leads[leadIdx].length > 0) {
          console.log(`Lead ${lead.id} (idx ${leadIdx}): Found ${selectedPatient.leads[leadIdx].length} beats`);
          
          // Use the real raw signal data if available
          const firstBeat = selectedPatient.leads[leadIdx][0];
          let signalData: number[] = [];
          
          if (firstBeat.rawSignal && firstBeat.rawSignal.length > 0) {
            // Use actual ECG signal - determine if we need to repeat
            const signalLength = firstBeat.rawSignal.length;
            
            // Intelligent repeat based on signal length
            // If signal is already long (full 10-second recording), repeat less
            let repeatCount = 1;
            if (signalLength < 500) {
              // Very short segment, repeat many times
              repeatCount = 30;
            } else if (signalLength < 2000) {
              // Medium segment, repeat moderate amount
              repeatCount = 10;
            } else if (signalLength < 5000) {
              // Long segment, repeat a few times
              repeatCount = 3;
            } else {
              // Very long segment (full 10-sec), minimal repeat
              repeatCount = 2;
            }
            
            for (let i = 0; i < repeatCount; i++) {
              signalData.push(...firstBeat.rawSignal);
            }
            console.log(`Lead ${lead.id}: Real signal ${signalLength} samples x ${repeatCount} = ${signalData.length} total`);
          } else {
            // Fallback: Generate from features if no raw signal
            console.log(`Lead ${lead.id}: No raw signal, generating from features`);
            selectedPatient.leads[leadIdx].forEach((beat) => {
              const waveform = generateECGFromFeatures(beat.features, 500, lead.id);
              signalData.push(...waveform);
            });
          }
          
          console.log(`Lead ${lead.id}: Total samples=${signalData.length}, min=${Math.min(...signalData)}, max=${Math.max(...signalData)}`);
          
          return {
            ...lead,
            data: signalData,
            analysis: selectedPatient.leads[leadIdx][0].label,
            severity: (selectedPatient.leads[leadIdx][0].label.includes('Normal') ? 'normal' : 'warning') as 'normal' | 'warning' | 'critical'
          };
        }
        
        console.warn(`Lead ${lead.id} has no data`);
        return lead;
      });

      console.log('Updated leads:', updatedLeads.map(l => ({ id: l.id, dataLength: l.data.length })));
      setLeads(updatedLeads);
      setIsInitialized(true);
    } else if (!useRealData || !selectedPatient) {
      setIsInitialized(false);
    }
  }, [useRealData, selectedPatient, isInitialized]);

  useEffect(() => {
    if (!isRecording) return;

    const interval = setInterval(() => {
      if (useRealData && selectedPatient) {
        // Scroll through the pre-loaded patient data
        const leadMapping: { [key: string]: number } = {
          'I': 0, 'II': 1, 'III': 2,
          'aVR': 3, 'aVL': 4, 'aVF': 5,
          'V1': 6, 'V2': 7, 'V3': 8,
          'V4': 9, 'V5': 10, 'V6': 11
        };

        leads.forEach(lead => {
          const leadIdx = leadMapping[lead.id];
          if (selectedPatient.leads[leadIdx] && selectedPatient.leads[leadIdx].length > 0) {
            // Cycle through beats
            const currentBeat = Math.floor(lead.data.length / 500) % selectedPatient.leads[leadIdx].length;
            const nextBeatFeatures = selectedPatient.leads[leadIdx][currentBeat].features;
            const waveform = generateECGFromFeatures(nextBeatFeatures);
            
            // Add next sample
            const sampleIdx = lead.data.length % 500;
            const newSample = waveform[sampleIdx] * amplitude;
            
            const maxDataPoints = 2500;
            updateLeadData(lead.id, [...lead.data.slice(-maxDataPoints + 1), newSample]);
          }
        });
      } else {
        // Use synthetic data
        leads.forEach(lead => {
          const newSample = generator.generateNextSample(lead.id);
          const maxDataPoints = 2500;
          updateLeadData(lead.id, [...lead.data.slice(-maxDataPoints), newSample]);
        });
      }

      // Update heart rate periodically
      if (leads[1].data.length > 500) {
        const hr = ECGGenerator.calculateHeartRate(leads[1].data.slice(-500));
        if (hr > 0) setHeartRate(hr);
      }
    }, 10 / speed);

    return () => clearInterval(interval);
  }, [isRecording, speed, leads, generator, updateLeadData, setHeartRate, selectedPatient, useRealData, amplitude]);

  useEffect(() => {
    generator.updateParameters({ amplitude });
  }, [amplitude, generator]);

  const handleToggleAnimation = () => {
    const newState = !isAnimating;
    setAnimating(newState);
    setRecording(newState);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">12-Lead ECG Monitoring</h2>
        
        {/* Control Buttons */}
        <div className="flex items-center space-x-2">
          {/* Settings Button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setShowSettings(!showSettings)}
            className={`flex items-center justify-center w-12 h-12 rounded-full shadow-lg transition-all duration-300 ${
              showSettings
                ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700'
                : 'bg-gradient-to-r from-gray-400 to-gray-500 hover:from-gray-500 hover:to-gray-600'
            }`}
            title="Settings"
          >
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </motion.button>
          
          {/* Play/Pause Button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={handleToggleAnimation}
            className={`flex items-center justify-center w-12 h-12 rounded-full shadow-lg transition-all duration-300 ${
              isAnimating 
                ? 'bg-gradient-to-r from-orange-400 to-red-500 hover:from-orange-500 hover:to-red-600' 
                : 'bg-gradient-to-r from-green-400 to-emerald-500 hover:from-green-500 hover:to-emerald-600'
            }`}
            title={isAnimating ? 'Pause ECG Motion' : 'Resume ECG Motion'}
          >
            {isAnimating ? (
              <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </motion.button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex gap-6">
        {/* Settings Panel - Collapsible */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 'auto', opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="w-80 bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8 border border-gray-200">
                <div className="flex items-center space-x-3 mb-8">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
                    Settings
                  </h3>
                </div>
                
                <div className="space-y-8">
                  {/* Speed Control */}
                  <div className="relative">
                    <div className="flex justify-between mb-4">
                      <label className="text-base font-semibold text-gray-700 flex items-center space-x-2">
                        <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        <span>Speed</span>
                      </label>
                      <motion.span 
                        key={speed}
                        initial={{ scale: 1.2 }}
                        animate={{ scale: 1 }}
                        className="text-base font-bold text-blue-600 bg-blue-50 px-4 py-1 rounded-full"
                      >
                        {speed}x
                      </motion.span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="3"
                      step="0.5"
                      value={speed}
                      onChange={(e) => setSpeed(parseFloat(e.target.value))}
                      className="w-full h-3 bg-gradient-to-r from-blue-100 to-blue-200 rounded-full appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((speed - 0.5) / 2.5) * 100}%, #dbeafe ${((speed - 0.5) / 2.5) * 100}%, #dbeafe 100%)`
                      }}
                    />
                    <div className="flex justify-between text-sm font-medium text-gray-500 mt-3">
                      <span className="bg-gray-100 px-3 py-1.5 rounded">0.5x</span>
                      <span className="bg-gray-100 px-3 py-1.5 rounded">3x</span>
                    </div>
                  </div>

                  {/* Amplitude Control */}
                  <div className="relative">
                    <div className="flex justify-between mb-4">
                      <label className="text-base font-semibold text-gray-700 flex items-center space-x-2">
                        <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                        </svg>
                        <span>Amplitude</span>
                      </label>
                      <motion.span 
                        key={amplitude}
                        initial={{ scale: 1.2 }}
                        animate={{ scale: 1 }}
                        className="text-base font-bold text-purple-600 bg-purple-50 px-4 py-1 rounded-full"
                      >
                        {amplitude}x
                      </motion.span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2"
                      step="0.1"
                      value={amplitude}
                      onChange={(e) => setAmplitude(parseFloat(e.target.value))}
                      className="w-full h-3 bg-gradient-to-r from-purple-100 to-purple-200 rounded-full appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, #9333ea 0%, #9333ea ${((amplitude - 0.5) / 1.5) * 100}%, #f3e8ff ${((amplitude - 0.5) / 1.5) * 100}%, #f3e8ff 100%)`
                      }}
                    />
                    <div className="flex justify-between text-sm font-medium text-gray-500 mt-3">
                      <span className="bg-gray-100 px-3 py-1.5 rounded">0.5x</span>
                      <span className="bg-gray-100 px-3 py-1.5 rounded">2x</span>
                    </div>
                  </div>

                  {/* ECG Metrics */}
                  <div className="border-t border-gray-300 pt-8">
                    <div className="flex items-center space-x-2 mb-4">
                      <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      <h4 className="text-base font-bold text-gray-700">ECG Metrics</h4>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center bg-white rounded-lg p-3 shadow-sm">
                        <span className="text-sm text-gray-600">Available Samples</span>
                        <span className="text-base font-bold text-gray-800">{leads[0]?.data.length || 0}</span>
                      </div>
                      <div className="flex justify-between items-center bg-white rounded-lg p-3 shadow-sm">
                        <span className="text-sm text-gray-600">Recording Time</span>
                        <span className="text-base font-bold text-gray-800">{isRecording ? recordingTime.toFixed(1) : '0.0'}s</span>
                      </div>
                      <div className="flex justify-between items-center bg-white rounded-lg p-3 shadow-sm">
                        <span className="text-sm text-gray-600">Sample Rate</span>
                        <span className="text-base font-bold text-gray-800">500 Hz</span>
                      </div>
                      <div className="flex justify-between items-center bg-white rounded-lg p-3 shadow-sm">
                        <span className="text-sm text-gray-600">Total Duration</span>
                        <span className="text-base font-bold text-gray-800">{((leads[0]?.data.length || 0) / 500).toFixed(1)}s</span>
                      </div>
                    </div>
                  </div>

                  {/* Denoising Panel */}
                  <DenoisingPanel isOpen={showSettings} selectedPatient={selectedPatient} />


                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ECG Grid - 12 Leads */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {leads.map((lead, index) => (
          <motion.div
            key={lead.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="bg-gray-50 rounded-lg p-4 border border-gray-200 shadow-sm"
          >
            <div className="mb-2">
              <h3 className="text-lg font-semibold" style={{ color: lead.color }}>
                {lead.name}
              </h3>
            </div>
            
            <ECGWaveform
              data={lead.data}
              leadName={lead.id}
              color={lead.color}
              animate={isAnimating}
              speed={speed}
              amplitude={amplitude}
              highlighted={highlightedLeads.includes(lead.id)}
            />
          </motion.div>
        ))}
        </div>
      </div>
    </div>
  );
};

export default ECGDashboard;
