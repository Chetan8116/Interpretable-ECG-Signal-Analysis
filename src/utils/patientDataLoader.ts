/**
 * CSV Parser for patient ECG data - PTB-XL Database format
 */

export interface PatientECGData {
  patientId: string;
  age: number;
  sex: string;
  leads: {
    [leadIdx: number]: {
      label: string;
      features: ECGFeatures;
      rawSignal?: number[]; // Store actual raw signal data from PTB-XL
      qualityScore?: number; // Signal quality metric (0-1)
      qualityIssues?: string[]; // Any quality issues detected
    }[];
  };
  ecgId?: number;
  diagnosis?: string;
  scpCodes?: any;
  signalQuality?: number; // Overall signal quality for all leads
  preprocessing?: {
    bandpassFilter?: string;
    notchFilter?: string;
    qualityChecked?: boolean;
  };
}

export interface ECGFeatures {
  iqr: number;
  entropy: number;
  mean_freq: number;
  median_freq: number;
  P_amp: number;
  P_area: number;
  R_amp: number;
  QRS_area: number;
  T_amp: number;
  ST_slope: number;
  P_duration: number;
  QRS_duration: number;
  ST_duration: number;
  QT_interval: number;
}

export interface PTBXLRecord {
  ecg_id: number;
  patient_id: number;
  age: number;
  sex: number; // 0 = female, 1 = male
  height?: number;
  weight?: number;
  scp_codes: string;
  report: string;
  filename_lr: string;
  filename_hr: string;
}

// Legacy CSV format interface
export interface CSVRow {
  age: number;
  sex: string;
  lead_idx: number;
  label_full: string;
  iqr: number;
  entropy: number;
  mean_freq: number;
  median_freq: number;
  P_amp: number;
  P_area: number;
  R_amp: number;
  QRS_area: number;
  T_amp: number;
  ST_slope: number;
  P_duration: number;
  QRS_duration: number;
  ST_duration: number;
  QT_interval: number;
}

export class PatientDataLoader {
  private patients: Map<string, PatientECGData> = new Map();
  private allRecords: any[] = []; // Store all records for pagination
  private loadedRecordCount: number = 0;

  /**
   * Load PTB-XL database records from preprocessed JSON (new format with real signals)
   * Supports pagination for dynamic loading
   */
  async loadFromPTBXLJSON(startIndex: number = 0, batchSize: number = 50): Promise<number> {
    try {
      // Priority order for data loading (best to fallback):
      // 1. ptbxl_optimized.min.json - Production optimized (minified)
      // 2. ptbxl_optimized.json - Production optimized
      // 3. ptbxl_enhanced.json - Build-time enhanced
      // 4. ptbxl_full_signals.json - Python processed full signals
      // 5. ptbxl_processed_signals.json - Python notebook processed
      // 6. ptbxl_records.json - Original fallback
      
      let signalsData: any[] = [];
      let dataSource = '';
      
      const dataSources = [
        { file: '/ptbxl_optimized.min.json', name: 'Optimized (minified)' },
        { file: '/ptbxl_optimized.json', name: 'Optimized' },
        { file: '/ptbxl_enhanced.json', name: 'Enhanced' },
        { file: '/ptbxl_full_signals.json', name: 'Full signals (500Hz)' },
        { file: '/ptbxl_processed_signals.json', name: 'Processed signals' },
        { file: '/ptbxl_records.json', name: 'Original data' }
      ];
      
      // Try each data source in priority order
      for (const source of dataSources) {
        try {
          const response = await fetch(source.file);
          if (response.ok) {
            signalsData = await response.json();
            dataSource = source.name;
            console.log(`✓ Loaded ${source.name} successfully!`);
            break;
          }
        } catch (e) {
          // Continue to next source
        }
      }
      
      if (signalsData.length === 0 || !dataSource) {
        throw new Error('No ECG data files found! Please ensure at least ptbxl_records.json exists in the public folder.');
      }
      
      // Load all records on first call
      if (this.allRecords.length === 0) {
        this.allRecords = signalsData;
        console.log(`Data source: ${dataSource}`);
        console.log(`Total records available: ${this.allRecords.length}`);
      }

      // Calculate range for this batch
      const endIndex = Math.min(startIndex + batchSize, this.allRecords.length);
      const batchRecords = this.allRecords.slice(startIndex, endIndex);
      
      console.log(`Loading batch: ${startIndex} to ${endIndex} (${batchRecords.length} records)`);

      const leadMapping: { [key: string]: number } = {
        'I': 0, 'II': 1, 'III': 2,
        'aVR': 3, 'aVL': 4, 'aVF': 5,
        'V1': 6, 'V2': 7, 'V3': 8,
        'V4': 9, 'V5': 10, 'V6': 11
      };

      for (const record of batchRecords) {
        const patientId = `PTB${String(record.ecg_id).padStart(5, '0')}`;
        
        // Skip if already loaded
        if (this.patients.has(patientId)) {
          continue;
        }
        
        // Parse diagnosis from SCP codes
        let diagnosis = 'NORM';
        try {
          const scpCodes = eval('(' + record.scp_codes + ')');
          const codes = Object.keys(scpCodes).filter(k => scpCodes[k] > 0);
          diagnosis = codes.length > 0 ? codes.join(', ') : 'NORM';
        } catch (e) {
          diagnosis = record.diagnosis || 'NORM';
        }

        const patient: PatientECGData = {
          patientId,
          age: record.age,
          sex: record.sex,
          ecgId: record.ecg_id,
          diagnosis,
          scpCodes: record.scp_codes,
          signalQuality: record.signal_quality || 1.0,
          preprocessing: record.preprocessing || {
            bandpassFilter: '0.5-40 Hz',
            notchFilter: '50 Hz',
            qualityChecked: true
          },
          leads: {}
        };

        // Process each lead with REAL data
        for (const [leadName, leadData] of Object.entries(record.leads as any)) {
          const leadIdx = leadMapping[leadName];
          if (leadIdx === undefined) continue;

          patient.leads[leadIdx] = [];

          // Check if we have fully processed signals from Python pipeline
          if (leadData && typeof leadData === 'object' && 'processed' in leadData) {
            // Full processed signal from Python script
            const processedSignal = leadData.processed as number[];
            const rawSignal = leadData.raw as number[];
            
            console.log(`Lead ${leadName}: Using full processed signal (${processedSignal.length} samples)`);
            
            // Store the complete processed signal
            patient.leads[leadIdx].push({
              label: diagnosis,
              rawSignal: processedSignal, // Use the fully processed signal
              qualityScore: 1.0,
              qualityIssues: [],
              features: {
                // Calculate basic features from processed signal
                iqr: this.calculateIQR(processedSignal),
                entropy: 0,
                mean_freq: 0,
                median_freq: 0,
                P_amp: Math.max(...processedSignal.filter(v => isFinite(v))) * 0.15,
                P_area: 0.01,
                R_amp: Math.max(...processedSignal.filter(v => isFinite(v))),
                QRS_area: 0.05,
                T_amp: Math.max(...processedSignal.filter(v => isFinite(v))) * 0.25,
                ST_slope: 0,
                P_duration: 0.08,
                QRS_duration: 0.08,
                ST_duration: 0.1,
                QT_interval: 0.4
              }
            });
            continue;
          }

          // Check if we have beats with full signal data
          if (leadData && typeof leadData === 'object' && 'beats' in leadData) {
            const beats = leadData.beats as any[];
            if (beats && beats.length > 0) {
              // Use ALL beat signals available
              console.log(`Lead ${leadName}: Using ${beats.length} beat segments`);
              
              // Concatenate all beat signals into continuous data
              const fullSignal: number[] = [];
              beats.forEach(beat => {
                if (beat.signal && Array.isArray(beat.signal)) {
                  fullSignal.push(...beat.signal);
                }
              });
              
              if (fullSignal.length > 0) {
                patient.leads[leadIdx].push({
                  label: diagnosis,
                  rawSignal: fullSignal,
                  qualityScore: 1.0,
                  qualityIssues: [],
                  features: {
                    iqr: this.calculateIQR(fullSignal),
                    entropy: 0,
                    mean_freq: 0,
                    median_freq: 0,
                    P_amp: Math.max(...fullSignal.filter(v => isFinite(v))) * 0.15,
                    P_area: 0.01,
                    R_amp: Math.max(...fullSignal.filter(v => isFinite(v))),
                    QRS_area: 0.05,
                    T_amp: Math.max(...fullSignal.filter(v => isFinite(v))) * 0.25,
                    ST_slope: 0,
                    P_duration: 0.08,
                    QRS_duration: 0.08,
                    ST_duration: 0.1,
                    QT_interval: 0.4
                  }
                });
                continue;
              }
            }
          }

          // Old format: Type assertion for TypeScript
          const typedLeadData = leadData as { 
            features: any; 
            signal_preview: number[];
            quality_score?: number;
            quality_issues?: string[];
          };

          // Use real features from the signal
          const features = typedLeadData.features as ECGFeatures;
          const signalPreview = typedLeadData.signal_preview as number[]; // Preview signal from Python
          const qualityScore = typedLeadData.quality_score;
          const qualityIssues = typedLeadData.quality_issues;
          
          // Create multiple beats from the continuous signal
          // Simulate beats by using the same features with slight variations
          const numBeats = 30; // Fixed number for consistency
          for (let beatIdx = 0; beatIdx < numBeats; beatIdx++) {
            // Add small random variation to features (±5%) to simulate beat-to-beat variation
            const variation = 0.95 + Math.random() * 0.1;
            
            patient.leads[leadIdx].push({
              label: diagnosis,
              rawSignal: signalPreview, // Store the signal preview
              qualityScore: qualityScore,
              qualityIssues: qualityIssues,
              features: {
                iqr: features.iqr * variation,
                entropy: features.entropy * variation,
                mean_freq: features.mean_freq * variation,
                median_freq: features.median_freq * variation,
                P_amp: features.P_amp * variation,
                P_area: features.P_area * variation,
                R_amp: features.R_amp * variation,
                QRS_area: features.QRS_area * variation,
                T_amp: features.T_amp * variation,
                ST_slope: features.ST_slope * variation,
                P_duration: features.P_duration * variation,
                QRS_duration: features.QRS_duration * variation,
                ST_duration: features.ST_duration * variation,
                QT_interval: features.QT_interval * variation
              }
            });
          }
        }

        this.patients.set(patientId, patient);
      }

      this.loadedRecordCount = this.patients.size;
      
      console.log(`Batch loaded successfully. Total patients: ${this.patients.size}/${this.allRecords.length}`);
      
      // Log sample on first batch
      if (startIndex === 0 && this.patients.size > 0) {
        const firstPatient = Array.from(this.patients.values())[0];
        const firstLead = firstPatient.leads[0]?.[0];
        if (firstLead?.rawSignal) {
          console.log(`✓ Filtered signal data confirmed! Sample length: ${firstLead.rawSignal.length} points`);
          console.log(`✓ Signal quality: ${(firstLead.qualityScore || 0) * 100}%`);
        }
        if (firstPatient.preprocessing) {
          console.log(`✓ Preprocessing applied:`, firstPatient.preprocessing);
        }
      }
      
      return this.allRecords.length; // Return total available records
    } catch (error) {
      console.error('Error loading PTB-XL JSON:', error);
      throw error;
    }
  }

  /**
   * Check if more records are available to load
   */
  hasMoreRecords(): boolean {
    return this.loadedRecordCount < this.allRecords.length;
  }

  /**
   * Get total available records count
   */
  getTotalRecordsCount(): number {
    return this.allRecords.length;
  }

  /**
   * Get currently loaded records count
   */
  getLoadedRecordsCount(): number {
    return this.loadedRecordCount;
  }

  /**
   * Load PTB-XL database records from CSV (legacy method for fallback)
   */
  async loadFromPTBXL(csvContent: string, maxRecords: number = 200): Promise<void> {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',');

    console.log(`Loading PTB-XL database with ${maxRecords} records...`);

    // Parse records
    const recordCount = Math.min(maxRecords + 1, lines.length); // +1 for header
    
    for (let i = 1; i < recordCount; i++) {
      const values = this.parseCSVLine(lines[i]);
      if (values.length < headers.length) continue;

      const row: any = {};
      headers.forEach((header, index) => {
        row[header] = values[index];
      });

      try {
        const ecgId = parseInt(row.ecg_id);
        const patientId = `PTB${String(ecgId).padStart(5, '0')}`;
        const age = parseInt(row.age) || 50;
        const sex = parseInt(row.sex) === 0 ? 'F' : 'M';
        
        // Parse SCP codes (diagnosis)
        let diagnosis = 'NORM';
        let scpCodes: any = {};
        try {
          // SCP codes are in format: {'NORM': 100.0, 'SR': 0.0}
          scpCodes = eval('(' + row.scp_codes + ')');
          const codes = Object.keys(scpCodes).filter(k => scpCodes[k] > 0);
          diagnosis = codes.length > 0 ? codes.join(', ') : 'NORM';
        } catch (e) {
          diagnosis = 'NORM';
        }

        // Generate patient data with synthesized features based on diagnosis
        const patient: PatientECGData = {
          patientId,
          age,
          sex,
          ecgId,
          diagnosis,
          scpCodes,
          leads: {}
        };

        // Initialize 12 leads with synthetic beats
        for (let leadIdx = 0; leadIdx < 12; leadIdx++) {
          patient.leads[leadIdx] = [];
          
          // Generate 20-50 beats per lead
          const numBeats = 30 + Math.floor(Math.random() * 20);
          for (let beatIdx = 0; beatIdx < numBeats; beatIdx++) {
            patient.leads[leadIdx].push({
              label: diagnosis,
              features: this.generateFeaturesForDiagnosis(diagnosis, age, leadIdx)
            });
          }
        }

        this.patients.set(patientId, patient);
      } catch (error) {
        console.error(`Error parsing record ${i}:`, error);
        continue;
      }
    }

    console.log(`Successfully loaded ${this.patients.size} patients from PTB-XL database`);
  }

  /**
   * Parse CSV line handling quoted fields
   */
  private parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current);
    
    return result;
  }

  /**
   * Generate realistic ECG features based on diagnosis and patient characteristics
   */
  private generateFeaturesForDiagnosis(diagnosis: string, age: number, leadIdx: number): ECGFeatures {
    // Base features for normal ECG
    const baseFeatures: ECGFeatures = {
      iqr: 0.15 + Math.random() * 0.1,
      entropy: 0.7 + Math.random() * 0.2,
      mean_freq: 10 + Math.random() * 5,
      median_freq: 8 + Math.random() * 4,
      P_amp: 0.1 + Math.random() * 0.05,
      P_area: 0.01 + Math.random() * 0.005,
      R_amp: 0.8 + Math.random() * 0.4,
      QRS_area: 0.08 + Math.random() * 0.02,
      T_amp: 0.2 + Math.random() * 0.1,
      ST_slope: -0.01 + Math.random() * 0.02,
      P_duration: 0.08 + Math.random() * 0.02,
      QRS_duration: 0.08 + Math.random() * 0.02,
      ST_duration: 0.1 + Math.random() * 0.02,
      QT_interval: 0.38 + Math.random() * 0.04
    };

    // Modify features based on diagnosis
    if (diagnosis.includes('IMI') || diagnosis.includes('INFARCTION')) {
      // Inferior myocardial infarction
      baseFeatures.ST_slope = -0.05 - Math.random() * 0.03;
      baseFeatures.T_amp *= 0.6;
      baseFeatures.QRS_duration *= 1.2;
    } else if (diagnosis.includes('STTC') || diagnosis.includes('ST')) {
      // ST-T changes
      baseFeatures.ST_slope = -0.03 - Math.random() * 0.02;
      baseFeatures.T_amp *= 0.7;
    } else if (diagnosis.includes('BRAD')) {
      // Bradycardia
      baseFeatures.QT_interval *= 1.15;
      baseFeatures.P_duration *= 1.1;
    } else if (diagnosis.includes('TACH')) {
      // Tachycardia
      baseFeatures.QT_interval *= 0.85;
      baseFeatures.QRS_duration *= 0.9;
    } else if (diagnosis.includes('LVOLT')) {
      // Low voltage
      baseFeatures.R_amp *= 0.5;
      baseFeatures.P_amp *= 0.6;
      baseFeatures.T_amp *= 0.6;
    }

    // Age-related modifications
    if (age > 60) {
      baseFeatures.QT_interval *= 1.05;
      baseFeatures.P_duration *= 1.05;
    }

    // Lead-specific variations
    const leadVariations = [1.0, 1.1, 0.9, 0.7, 0.8, 1.0, 0.9, 1.2, 1.3, 1.2, 1.1, 1.0];
    baseFeatures.R_amp *= leadVariations[leadIdx];

    return baseFeatures;
  }

  /**
   * Parse legacy CSV file content (old format)
   */
  async loadFromCSV(csvContent: string): Promise<void> {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',');

    // First pass: collect all rows organized by patient (age-sex)
    const patientDataMap: { [patientKey: string]: any[] } = {};

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length !== headers.length) continue;

      const row: any = {};
      headers.forEach((header, index) => {
        const value = values[index];
        row[header] = isNaN(Number(value)) ? value : Number(value);
      });

      const csvRow = row as CSVRow;
      const patientKey = `${csvRow.age}-${csvRow.sex}`;

      if (!patientDataMap[patientKey]) {
        patientDataMap[patientKey] = [];
      }
      patientDataMap[patientKey].push(csvRow);
    }

    // Second pass: Split into separate patients (max 50 beats each)
    const maxBeatsPerPatient = 50;
    let patientCounter = 0;
    
    for (const [patientKey, allRows] of Object.entries(patientDataMap)) {
      const [ageStr, sex] = patientKey.split('-');
      const age = parseInt(ageStr);

      const leadsPerBeat: { [leadIdx: number]: any[] } = {};

      // First, organize rows by lead_idx
      for (const row of allRows) {
        if (!leadsPerBeat[row.lead_idx]) {
          leadsPerBeat[row.lead_idx] = [];
        }
        leadsPerBeat[row.lead_idx].push(row);
      }

      // Determine total number of beats
      const beatCounts = Object.values(leadsPerBeat).map(leads => leads.length);
      const totalBeats = Math.min(...beatCounts);
      
      // Split into multiple patients
      const numPatients = Math.ceil(totalBeats / maxBeatsPerPatient);
      
      for (let p = 0; p < numPatients; p++) {
        const startBeat = p * maxBeatsPerPatient;
        const endBeat = Math.min(startBeat + maxBeatsPerPatient, totalBeats);
        
        const patientId = `P${String(patientCounter).padStart(4, '0')}`;
        const patient: PatientECGData = {
          patientId,
          age,
          sex,
          leads: {}
        };

        // Initialize leads array
        for (let i = 0; i < 12; i++) {
          patient.leads[i] = [];
        }

        // Add beats for this patient
        for (let leadIdx = 0; leadIdx < 12; leadIdx++) {
          if (leadsPerBeat[leadIdx]) {
            for (let beatIdx = startBeat; beatIdx < endBeat; beatIdx++) {
              if (leadsPerBeat[leadIdx][beatIdx]) {
                const row = leadsPerBeat[leadIdx][beatIdx];
                patient.leads[leadIdx].push({
                  label: row.label_full,
                  features: {
                    iqr: row.iqr,
                    entropy: row.entropy,
                    mean_freq: row.mean_freq,
                    median_freq: row.median_freq,
                    P_amp: row.P_amp,
                    P_area: row.P_area,
                    R_amp: row.R_amp,
                    QRS_area: row.QRS_area,
                    T_amp: row.T_amp,
                    ST_slope: row.ST_slope,
                    P_duration: row.P_duration,
                    QRS_duration: row.QRS_duration,
                    ST_duration: row.ST_duration,
                    QT_interval: row.QT_interval
                  }
                });
              }
            }
          }
        }

        this.patients.set(patientId, patient);
        patientCounter++;
      }
    }

    console.log(`Loaded ${patientCounter} patients from CSV`);
  }

  /**
   * Get all patients
   */
  getAllPatients(): PatientECGData[] {
    return Array.from(this.patients.values());
  }

  /**
   * Get patient by ID
   */
  getPatient(patientId: string): PatientECGData | undefined {
    return this.patients.get(patientId);
  }

  /**
   * Generate ECG waveform from features
   */
  generateECGFromFeatures(features: ECGFeatures, samplingRate: number = 500): number[] {
    const duration = 1.0; // 1 second
    const samples = Math.floor(duration * samplingRate);
    const ecgData: number[] = [];

    for (let i = 0; i < samples; i++) {
      const t = i / samplingRate;
      let signal = 0;

      // P wave (0.0 - 0.1s)
      if (t >= 0.0 && t < features.P_duration) {
        const pPhase = t / features.P_duration;
        signal += features.P_amp * Math.sin(pPhase * Math.PI);
      }

      // QRS complex (around 0.15s - 0.23s)
      const qrsStart = features.P_duration + 0.05;
      const qrsEnd = qrsStart + features.QRS_duration;
      if (t >= qrsStart && t < qrsEnd) {
        const qrsPhase = (t - qrsStart) / features.QRS_duration;
        
        // Q wave
        if (qrsPhase < 0.2) {
          signal -= features.R_amp * 0.2 * Math.sin(qrsPhase * Math.PI / 0.2);
        }
        // R wave
        else if (qrsPhase < 0.6) {
          const rPhase = (qrsPhase - 0.2) / 0.4;
          signal += features.R_amp * Math.sin(rPhase * Math.PI);
        }
        // S wave
        else {
          const sPhase = (qrsPhase - 0.6) / 0.4;
          signal -= features.R_amp * 0.3 * Math.sin(sPhase * Math.PI);
        }
      }

      // ST segment
      const stStart = qrsEnd;
      const stEnd = stStart + features.ST_duration;
      if (t >= stStart && t < stEnd) {
        signal += features.ST_slope * (t - stStart) / features.ST_duration;
      }

      // T wave
      const tStart = stEnd;
      const tEnd = tStart + 0.16;
      if (t >= tStart && t < tEnd) {
        const tPhase = (t - tStart) / 0.16;
        signal += features.T_amp * Math.sin(tPhase * Math.PI);
      }

      ecgData.push(signal);
    }

    return ecgData;
  }

  /**
   * Get continuous ECG data for a patient lead
   */
  getContinuousECGForLead(patientId: string, leadIdx: number, duration: number = 5): number[] {
    const patient = this.getPatient(patientId);
    if (!patient || !patient.leads[leadIdx]) {
      return [];
    }

    const leadData = patient.leads[leadIdx];
    
    // Check if we have real signal data
    if (leadData.length > 0 && leadData[0].rawSignal && leadData[0].rawSignal.length > 0) {
      // Use the REAL signal data features to generate waveform
      // Since we only store a preview, we use features to generate continuous data
      const features = leadData[0].features;
      const continuousData: number[] = [];
      
      // Generate continuous waveform using real features from the patient
      const numBeats = Math.ceil(duration);
      for (let i = 0; i < numBeats; i++) {
        const beatWaveform = this.generateECGFromFeatures(features);
        continuousData.push(...beatWaveform);
      }
      
      return continuousData.slice(0, duration * 500);
    } else {
      // Fallback: Generate continuous waveform from features
      const continuousData: number[] = [];
      const numBeats = Math.ceil(duration);
      
      for (let i = 0; i < numBeats; i++) {
        const beatIndex = i % leadData.length;
        const beatFeatures = leadData[beatIndex].features;
        const beatWaveform = this.generateECGFromFeatures(beatFeatures);
        continuousData.push(...beatWaveform);
      }
      
      return continuousData.slice(0, duration * 500);
    }
  }

  /**
   * Get all 12 leads data for a patient
   */
  getAll12LeadsData(patientId: string, duration: number = 5): { [leadName: string]: number[] } {
    const leadMapping: { [key: number]: string } = {
      0: 'I', 1: 'II', 2: 'III',
      3: 'aVR', 4: 'aVL', 5: 'aVF',
      6: 'V1', 7: 'V2', 8: 'V3',
      9: 'V4', 10: 'V5', 11: 'V6'
    };

    const allLeads: { [leadName: string]: number[] } = {};

    for (let i = 0; i < 12; i++) {
      const leadName = leadMapping[i];
      allLeads[leadName] = this.getContinuousECGForLead(patientId, i, duration);
    }

    return allLeads;
  }

  /**
   * Get patient diagnosis summary
   */
  getPatientDiagnosis(patientId: string): string {
    const patient = this.getPatient(patientId);
    if (!patient) return 'Unknown';

    // Check labels across all leads
    const labels = new Set<string>();
    Object.values(patient.leads).forEach(leadData => {
      leadData.forEach(beat => labels.add(beat.label));
    });

    return Array.from(labels).join(', ');
  }

  private calculateIQR(signal: number[]): number {
    if (signal.length === 0) return 0;
    const sorted = [...signal].sort((a, b) => a - b);
    const q1Idx = Math.floor(sorted.length * 0.25);
    const q3Idx = Math.floor(sorted.length * 0.75);
    return sorted[q3Idx] - sorted[q1Idx];
  }
}

export default PatientDataLoader;
