/**
 * ECG Signal Generator
 * Generates realistic synthetic ECG waveforms for all 12 leads
 */

export interface ECGParameters {
  heartRate: number;
  samplingRate: number;
  amplitude: number;
  noiseLevel: number;
  condition?: 'normal' | 'afib' | 'st_elevation' | 'pvc' | 'vtach';
}

export class ECGGenerator {
  private parameters: ECGParameters;
  private time: number = 0;
  
  // Standard ECG wave parameters (in seconds)
  private readonly WAVE_PARAMS = {
    p: { duration: 0.08, amplitude: 0.15 },
    qrs: { duration: 0.08, amplitude: 1.0 },
    t: { duration: 0.16, amplitude: 0.3 },
    pr_interval: 0.16,
    qt_interval: 0.4,
  };

  constructor(parameters: Partial<ECGParameters> = {}) {
    this.parameters = {
      heartRate: 75,
      samplingRate: 500, // Hz
      amplitude: 1.0,
      noiseLevel: 0.05,
      condition: 'normal',
      ...parameters
    };
  }

  /**
   * Generate a single cardiac cycle (PQRST complex)
   */
  private generateCardiacCycle(phase: number, leadModifier: number = 1): number {
    const { heartRate, amplitude } = this.parameters;
    const cycleLength = 60 / heartRate; // seconds per beat
    
    let signal = 0;
    
    // P wave (atrial depolarization)
    const pPhase = (phase % cycleLength) / cycleLength;
    if (pPhase < 0.15) {
      const pWave = Math.sin(pPhase * Math.PI / 0.15);
      signal += this.WAVE_PARAMS.p.amplitude * pWave;
    }
    
    // QRS complex (ventricular depolarization)
    if (pPhase >= 0.15 && pPhase < 0.25) {
      const qrsPhase = (pPhase - 0.15) / 0.1;
      
      // Q wave (negative deflection)
      if (qrsPhase < 0.2) {
        signal -= 0.2 * Math.sin(qrsPhase * Math.PI / 0.2);
      }
      // R wave (positive peak)
      else if (qrsPhase < 0.6) {
        const rPhase = (qrsPhase - 0.2) / 0.4;
        signal += this.WAVE_PARAMS.qrs.amplitude * Math.sin(rPhase * Math.PI);
      }
      // S wave (negative deflection)
      else {
        const sPhase = (qrsPhase - 0.6) / 0.4;
        signal -= 0.3 * Math.sin(sPhase * Math.PI);
      }
    }
    
    // T wave (ventricular repolarization)
    if (pPhase >= 0.35 && pPhase < 0.55) {
      const tPhase = (pPhase - 0.35) / 0.2;
      signal += this.WAVE_PARAMS.t.amplitude * Math.sin(tPhase * Math.PI);
    }
    
    // Apply amplitude and lead-specific modification
    signal *= amplitude * leadModifier;
    
    // Add baseline noise
    signal += (Math.random() - 0.5) * this.parameters.noiseLevel;
    
    return signal;
  }

  /**
   * Apply condition-specific modifications
   */
  private applyCondition(signal: number, phase: number): number {
    const { condition } = this.parameters;
    
    switch (condition) {
      case 'afib':
        // Irregular rhythm, absent P waves
        return signal + (Math.random() - 0.5) * 0.3;
      
      case 'st_elevation':
        // Elevated ST segment
        const cyclePhase = phase % (60 / this.parameters.heartRate);
        if (cyclePhase > 0.25 && cyclePhase < 0.35) {
          return signal + 0.3;
        }
        return signal;
      
      case 'pvc':
        // Premature ventricular contraction
        if (Math.random() < 0.1) {
          return signal * 1.5;
        }
        return signal;
      
      case 'vtach':
        // Ventricular tachycardia (rapid, wide QRS)
        return signal * 1.2;
      
      default:
        return signal;
    }
  }

  /**
   * Get lead-specific amplitude modifier
   */
  private getLeadModifier(leadName: string): number {
    const modifiers: { [key: string]: number } = {
      'I': 1.0,
      'II': 1.2,
      'III': 0.8,
      'aVR': -0.5,
      'aVL': 0.6,
      'aVF': 1.0,
      'V1': 0.7,
      'V2': 1.5,
      'V3': 1.8,
      'V4': 1.6,
      'V5': 1.2,
      'V6': 1.0,
    };
    
    return modifiers[leadName] || 1.0;
  }

  /**
   * Generate ECG data for a specific lead
   */
  generateLead(leadName: string, duration: number = 5): number[] {
    const { samplingRate, heartRate } = this.parameters;
    const samples = Math.floor(duration * samplingRate);
    const data: number[] = [];
    const leadModifier = this.getLeadModifier(leadName);
    
    for (let i = 0; i < samples; i++) {
      const time = i / samplingRate;
      let signal = this.generateCardiacCycle(time, leadModifier);
      signal = this.applyCondition(signal, time);
      data.push(signal);
    }
    
    return data;
  }

  /**
   * Generate continuous ECG data (for real-time streaming)
   */
  generateNextSample(leadName: string): number {
    const { samplingRate } = this.parameters;
    const leadModifier = this.getLeadModifier(leadName);
    
    let signal = this.generateCardiacCycle(this.time, leadModifier);
    signal = this.applyCondition(signal, this.time);
    
    this.time += 1 / samplingRate;
    
    return signal;
  }

  /**
   * Generate all 12 leads simultaneously
   */
  generateAll12Leads(duration: number = 5): { [key: string]: number[] } {
    const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
    const data: { [key: string]: number[] } = {};
    
    leads.forEach(lead => {
      data[lead] = this.generateLead(lead, duration);
    });
    
    return data;
  }

  /**
   * Calculate heart rate from ECG data (R-peak detection)
   */
  static calculateHeartRate(ecgData: number[], samplingRate: number = 500): number {
    const threshold = Math.max(...ecgData) * 0.6;
    const peaks: number[] = [];
    
    for (let i = 1; i < ecgData.length - 1; i++) {
      if (ecgData[i] > threshold && 
          ecgData[i] > ecgData[i - 1] && 
          ecgData[i] > ecgData[i + 1]) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1] > samplingRate * 0.3) {
          peaks.push(i);
        }
      }
    }
    
    if (peaks.length < 2) return 0;
    
    const rrIntervals = [];
    for (let i = 1; i < peaks.length; i++) {
      rrIntervals.push((peaks[i] - peaks[i - 1]) / samplingRate);
    }
    
    const avgRR = rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length;
    return Math.round(60 / avgRR);
  }

  /**
   * Reset time counter
   */
  reset(): void {
    this.time = 0;
  }

  /**
   * Update parameters
   */
  updateParameters(parameters: Partial<ECGParameters>): void {
    this.parameters = { ...this.parameters, ...parameters };
  }
}

export default ECGGenerator;
