import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { PatientECGData } from '../utils/patientDataLoader';

interface PatientSelectorProps {
  patients: PatientECGData[];
  selectedPatient: PatientECGData | null;
  onSelectPatient: (patient: PatientECGData) => void;
  onClose: () => void;
}

const PatientSelector: React.FC<PatientSelectorProps> = ({
  patients,
  selectedPatient,
  onSelectPatient,
  onClose,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 48;

  const filteredPatients = useMemo(() => {
    const query = searchTerm.toLowerCase().trim();
    if (!query) return patients;

    return patients.filter((patient) => {
      return (
        patient.patientId.toLowerCase().includes(query) ||
        patient.age.toString().includes(query) ||
        patient.sex.toLowerCase().includes(query)
      );
    });
  }, [patients, searchTerm]);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm]);

  const totalPages = Math.max(1, Math.ceil(filteredPatients.length / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentPatients = filteredPatients.slice(startIndex, endIndex);

  const visibleQualityCount = filteredPatients.filter((patient) => patient.signalQuality !== undefined).length;

  const goToPage = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-[radial-gradient(circle_at_top,rgba(15,118,110,0.28),transparent_35%),rgba(2,6,23,0.76)] p-3 backdrop-blur-xl sm:p-6"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, y: 28, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 24, scale: 0.98 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto flex h-full w-full max-w-[1480px] flex-col overflow-hidden rounded-[2rem] border border-white/15 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(244,250,251,0.94))] shadow-[0_40px_120px_rgba(2,6,23,0.42)]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-slate-200/80 bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.16),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.96),rgba(248,250,252,0.88))] px-5 py-5 sm:px-8 sm:py-7">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-white/80 px-4 py-2 text-xs font-semibold uppercase tracking-[0.28em] text-teal-800 shadow-sm">
                <span className="h-2 w-2 rounded-full bg-teal-500" />
                Patient Navigator
              </div>
              <h2 className="mt-4 text-4xl font-semibold text-slate-950 sm:text-5xl">Select a patient record</h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                Browse PTB-XL entries, search by patient metadata, and jump directly into waveform review with a
                cleaner, more modern selection workspace.
              </p>
            </div>

            <div className="flex items-center gap-3 self-start xl:self-auto">
              <button
                onClick={onClose}
                className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-950 text-lg font-semibold text-white shadow-[0_14px_32px_rgba(15,23,42,0.28)] transition hover:scale-[1.03]"
                aria-label="Close patient selector"
              >
                ×
              </button>
            </div>
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-[1.3fr_0.7fr]">
            <div className="relative">
              <span className="pointer-events-none absolute left-6 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center text-slate-400">
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M17 10.5A6.5 6.5 0 114 10.5a6.5 6.5 0 0113 0z" />
                </svg>
              </span>
              <input
                type="text"
                placeholder="Search by patient ID, age, or sex"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full rounded-[1.2rem] border border-slate-200 bg-white/92 py-4 pl-[3.55rem] pr-5 text-[15px] leading-6 text-slate-800 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)] outline-none transition focus:border-teal-400 focus:ring-4 focus:ring-teal-100"
              />
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-[1.15rem] border border-white/70 bg-white/80 p-4 shadow-sm">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Loaded</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">{patients.length}</p>
              </div>
              <div className="rounded-[1.15rem] border border-white/70 bg-white/80 p-4 shadow-sm">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Filtered</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">{filteredPatients.length}</p>
              </div>
              <div className="rounded-[1.15rem] border border-white/70 bg-white/80 p-4 shadow-sm">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Quality Tags</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">{visibleQualityCount}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="min-h-0 flex-1 px-5 py-5 sm:px-8">
          <div className="custom-scrollbar grid max-h-full gap-4 overflow-y-auto pr-1 md:grid-cols-2 xl:grid-cols-3">
            {currentPatients.map((patient) => {
              const isSelected = selectedPatient?.patientId === patient.patientId;
              const numBeats = Object.values(patient.leads).reduce((sum, lead) => sum + lead.length, 0);
              const leadCount = Object.keys(patient.leads).length;
              const qualityValue = patient.signalQuality !== undefined ? `${(patient.signalQuality * 100).toFixed(0)}%` : 'N/A';
              const qualityTone =
                patient.signalQuality === undefined
                  ? 'text-slate-500 bg-slate-100'
                  : patient.signalQuality > 0.9
                    ? 'text-emerald-700 bg-emerald-100'
                    : patient.signalQuality > 0.7
                      ? 'text-amber-700 bg-amber-100'
                      : 'text-rose-700 bg-rose-100';

              return (
                <button
                  key={patient.patientId}
                  type="button"
                  onClick={() => onSelectPatient(patient)}
                  className={`group relative overflow-hidden rounded-[1.5rem] border p-5 text-left transition duration-200 ${
                    isSelected
                      ? 'border-teal-400 bg-[linear-gradient(180deg,rgba(240,253,250,0.96),rgba(236,254,255,0.92))] shadow-[0_20px_50px_rgba(20,184,166,0.18)]'
                      : 'border-slate-200 bg-white/88 shadow-[0_16px_40px_rgba(15,23,42,0.08)] hover:border-teal-200 hover:shadow-[0_20px_45px_rgba(15,23,42,0.12)]'
                  }`}
                >
                  <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(20,184,166,0.12),transparent_28%)] opacity-0 transition group-hover:opacity-100" />
                  <div className="relative">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-center gap-4">
                        <div
                          className={`flex h-14 w-14 items-center justify-center rounded-2xl text-lg font-bold shadow-sm ${
                            patient.sex === 'M'
                              ? 'bg-[linear-gradient(135deg,#dbeafe,#bfdbfe)] text-blue-700'
                              : 'bg-[linear-gradient(135deg,#fce7f3,#fbcfe8)] text-pink-700'
                          }`}
                        >
                          {patient.sex}
                        </div>
                        <div>
                          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Patient ID</p>
                          <h3 className="mt-1 text-lg font-semibold text-slate-950">{patient.patientId}</h3>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {isSelected && (
                          <span className="rounded-full bg-teal-600 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-white">
                            Active
                          </span>
                        )}
                        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${qualityTone}`}>{qualityValue}</span>
                      </div>
                    </div>

                    <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-slate-600">
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Age</p>
                        <p className="mt-1 text-base font-semibold text-slate-900">{patient.age} years</p>
                      </div>
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Sex</p>
                        <p className="mt-1 text-base font-semibold text-slate-900">{patient.sex === 'M' ? 'Male' : 'Female'}</p>
                      </div>
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Leads</p>
                        <p className="mt-1 text-base font-semibold text-slate-900">{leadCount}/12 available</p>
                      </div>
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Beats</p>
                        <p className="mt-1 text-base font-semibold text-slate-900">{numBeats}</p>
                      </div>
                    </div>

                    {patient.preprocessing && (
                      <div className="mt-4 flex flex-wrap gap-2">
                        <span className="rounded-full bg-cyan-50 px-3 py-1.5 text-xs font-medium text-cyan-700 ring-1 ring-cyan-100">
                          Bandpass: {patient.preprocessing.bandpassFilter}
                        </span>
                        <span className="rounded-full bg-violet-50 px-3 py-1.5 text-xs font-medium text-violet-700 ring-1 ring-violet-100">
                          Notch: {patient.preprocessing.notchFilter}
                        </span>
                      </div>
                    )}

                    <div className="mt-5 flex items-center justify-between border-t border-slate-200/70 pt-4">
                      <p className="text-sm text-slate-500">
                        {isSelected ? 'Current patient in workspace' : 'Open this patient in the analysis workspace'}
                      </p>
                      <span className="rounded-full bg-slate-950 px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-white transition group-hover:bg-teal-700">
                        Review
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}

            {filteredPatients.length === 0 && (
              <div className="col-span-full rounded-[1.5rem] border border-dashed border-slate-300 bg-white/70 px-6 py-16 text-center">
                <p className="text-2xl font-semibold text-slate-900">No patient records found</p>
                <p className="mt-2 text-sm text-slate-500">Try another patient ID, age, or sex filter.</p>
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-4 border-t border-slate-200/80 bg-white/82 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <p className="text-sm text-slate-600">
            Showing <span className="font-semibold text-slate-900">{filteredPatients.length === 0 ? 0 : startIndex + 1}</span>-
            <span className="font-semibold text-slate-900">{Math.min(endIndex, filteredPatients.length)}</span> of{' '}
            <span className="font-semibold text-slate-900">{filteredPatients.length}</span> patients
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage === 1}
              className="rounded-full border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Previous
            </button>
            <span className="rounded-full bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-800">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => goToPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="rounded-full bg-[linear-gradient(135deg,#0f766e,#0891b2)] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_12px_30px_rgba(8,145,178,0.25)] transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </motion.div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 12px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(226, 232, 240, 0.55);
          border-radius: 999px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #0f766e, #0891b2);
          border-radius: 999px;
          border: 2px solid rgba(255, 255, 255, 0.6);
        }
      `}</style>
    </motion.div>
  );
};

export default PatientSelector;
