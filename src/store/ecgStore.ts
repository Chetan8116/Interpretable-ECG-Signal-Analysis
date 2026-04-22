import { create } from 'zustand';
import { ECGPrediction } from '../models/MLPModel';
import { PatientECGData } from '../utils/patientDataLoader';

export interface LeadData {
  id: string;
  name: string;
  data: number[];
  color: string;
  analysis: string | null;
  severity: 'normal' | 'warning' | 'critical';
}

export interface ECGState {
  leads: LeadData[];
  isRecording: boolean;
  isAnimating: boolean;
  heartRate: number;
  analysisResults: ECGPrediction | null;
  currentTime: number;
  speed: number;
  amplitude: number;
  selectedPatient: PatientECGData | null;
  useRealData: boolean;
  highlightedLeads: string[];
  
  // Actions
  setLeads: (leads: LeadData[]) => void;
  updateLeadData: (leadId: string, data: number[]) => void;
  setRecording: (recording: boolean) => void;
  setAnimating: (animating: boolean) => void;
  setHeartRate: (rate: number) => void;
  setAnalysisResults: (results: ECGPrediction | null) => void;
  setCurrentTime: (time: number) => void;
  setSpeed: (speed: number) => void;
  setAmplitude: (amplitude: number) => void;
  setSelectedPatient: (patient: PatientECGData | null) => void;
  setUseRealData: (useReal: boolean) => void;
  setHighlightedLeads: (leads: string[]) => void;
  resetAll: () => void;
}

const LEAD_NAMES = [
  { id: 'I', name: 'Lead I', color: '#22c55e' },
  { id: 'II', name: 'Lead II', color: '#3b82f6' },
  { id: 'III', name: 'Lead III', color: '#a855f7' },
  { id: 'aVR', name: 'aVR', color: '#eab308' },
  { id: 'aVL', name: 'aVL', color: '#ec4899' },
  { id: 'aVF', name: 'aVF', color: '#14b8a6' },
  { id: 'V1', name: 'V1', color: '#f97316' },
  { id: 'V2', name: 'V2', color: '#06b6d4' },
  { id: 'V3', name: 'V3', color: '#8b5cf6' },
  { id: 'V4', name: 'V4', color: '#10b981' },
  { id: 'V5', name: 'V5', color: '#f59e0b' },
  { id: 'V6', name: 'V6', color: '#ef4444' },
];

const initialLeads: LeadData[] = LEAD_NAMES.map(lead => ({
  id: lead.id,
  name: lead.name,
  data: [],
  color: lead.color,
  analysis: null,
  severity: 'normal'
}));

export const useECGStore = create<ECGState>((set) => ({
  leads: initialLeads,
  isRecording: false,
  isAnimating: false,
  heartRate: 75,
  analysisResults: null,
  currentTime: 0,
  speed: 1,
  amplitude: 1,
  selectedPatient: null,
  useRealData: false,
  highlightedLeads: [],

  setLeads: (leads) => set({ leads }),
  
  updateLeadData: (leadId, data) => set((state) => ({
    leads: state.leads.map(lead =>
      lead.id === leadId ? { ...lead, data } : lead
    )
  })),

  setRecording: (recording) => set({ isRecording: recording }),
  
  setAnimating: (animating) => set({ isAnimating: animating }),
  
  setHeartRate: (rate) => set({ heartRate: rate }),
  
  setAnalysisResults: (results) => set({ analysisResults: results }),
  
  setCurrentTime: (time) => set({ currentTime: time }),
  
  setSpeed: (speed) => set({ speed }),
  
  setAmplitude: (amplitude) => set({ amplitude }),
  
  setSelectedPatient: (patient) => set({ selectedPatient: patient }),
  
  setUseRealData: (useReal) => set({ useRealData: useReal }),
  
  setHighlightedLeads: (leads) => set({ highlightedLeads: leads }),
  
  resetAll: () => set({
    leads: initialLeads,
    isRecording: false,
    isAnimating: false,
    heartRate: 75,
    analysisResults: null,
    currentTime: 0,
    speed: 1,
    amplitude: 1,
    highlightedLeads: []
  })
}));
