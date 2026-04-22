import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AnimatePresence,
  motion,
} from 'framer-motion';
import ECGDashboard from './components/ECGDashboard';
import PatientSelector from './components/PatientSelector';
import ClinicalFeaturesTable from './components/ClinicalFeaturesTable';
import SHAPExplainability from './components/SHAPExplainability';
import LeadWiseSHAP from './components/LeadWiseSHAP';
import ExpertDashboard from './components/ExpertDashboard';
import ECGHeatmapViewer from './components/ECGHeatmapViewer';
import SignalProcessingViewer from './components/SignalProcessingViewer';
import { useECGStore } from './store/ecgStore';
import PatientDataLoader, { PatientECGData } from './utils/patientDataLoader';

type SectionShellProps = {
  eyebrow: string;
  title: string;
  description: string;
  children: React.ReactNode;
};

const sectionCards = [
  {
    title: '12-Lead Monitoring',
    description: 'Live waveform layout with control over signal speed, amplitude, and patient playback.',
  },
  {
    title: 'Signal Cleanup',
    description: 'Inspect the denoising and filtering pipeline stage by stage before model interpretation.',
  },
  {
    title: 'Explainability',
    description: 'Track influential leads, abnormal features, and heatmap-backed clinical context.',
  },
];

const INITIAL_BATCH_SIZE = 100;
const LOAD_MORE_BATCH_SIZE = 100;

function SectionShell({ eyebrow, title, description, children }: SectionShellProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-[2rem] border border-white/55 bg-white/72 p-3 shadow-[0_28px_90px_rgba(15,23,42,0.14)] backdrop-blur-2xl"
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(34,197,94,0.12),transparent_30%),radial-gradient(circle_at_bottom_left,rgba(14,165,233,0.14),transparent_35%)]" />
      <div className="relative rounded-[1.6rem] border border-slate-200/65 bg-white/78 p-6 md:p-8 xl:p-10">
        <div className="mb-6 flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-[0.35em] text-teal-700/80">{eyebrow}</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-900 md:text-3xl">{title}</h2>
          </div>
          <p className="max-w-2xl text-sm leading-6 text-slate-600 xl:text-right">{description}</p>
        </div>
        {children}
      </div>
    </motion.section>
  );
}

function App() {
  const { isRecording, selectedPatient, setSelectedPatient, setUseRealData } = useECGStore();
  const [patients, setPatients] = useState<PatientECGData[]>([]);
  const [dataLoader] = useState(() => new PatientDataLoader());
  const [showPatientSelector, setShowPatientSelector] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState({ loaded: 0, total: 0 });
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const heroRef = useRef<HTMLElement | null>(null);
  const monitoringSectionRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const loadPatientsBatch = async (startIndex: number, batchSize: number) => {
      const totalRecords = await dataLoader.loadFromPTBXLJSON(startIndex, batchSize);
      const loadedPatients = dataLoader.getAllPatients();
      setPatients(loadedPatients);
      setLoadingProgress({ loaded: loadedPatients.length, total: totalRecords });
      return totalRecords;
    };

    const loadPTBXLData = async () => {
      try {
        console.log('Starting PTB-XL database load (real signals, dynamic loading)...');
        const totalRecords = await loadPatientsBatch(0, INITIAL_BATCH_SIZE);
        setIsLoading(false);
        console.log(
          `Initial batch loaded: ${dataLoader.getLoadedRecordsCount()}/${totalRecords} patients`,
        );
      } catch (error) {
        console.error('Error loading PTB-XL database:', error);
        setIsLoading(false);
      }
    };

    loadPTBXLData();
  }, [dataLoader]);

  const scrollToMonitoring = (extraOffset: number = 0) => {
    if (!monitoringSectionRef.current) {
      return;
    }

    const rect = monitoringSectionRef.current.getBoundingClientRect();
    const stickyHeaderOffset = 92;
    const top = window.scrollY + rect.top - stickyHeaderOffset + extraOffset;
    window.scrollTo({ top, behavior: 'smooth' });
  };

  const loadMorePatients = async () => {
    if (isLoadingMore || !dataLoader.hasMoreRecords()) {
      return;
    }

    setIsLoadingMore(true);
    try {
      const startIndex = dataLoader.getLoadedRecordsCount();
      const totalRecords = await dataLoader.loadFromPTBXLJSON(startIndex, LOAD_MORE_BATCH_SIZE);
      const updatedPatients = dataLoader.getAllPatients();
      setPatients([...updatedPatients]);
      setLoadingProgress({ loaded: updatedPatients.length, total: totalRecords });
      console.log(`Loaded more patients: ${updatedPatients.length}/${totalRecords}`);
    } catch (error) {
      console.error('Error loading more patients:', error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  const handleSelectPatient = (patient: PatientECGData) => {
    console.log('New patient selected:', patient.patientId);
    setSelectedPatient(patient);
    setUseRealData(true);
    setShowPatientSelector(false);

    window.setTimeout(() => {
      scrollToMonitoring(280);
    }, 180);
  };

  const stats = useMemo(
    () => [
      {
        label: 'Patients Ready',
        value: patients.length.toString(),
        canLoadMore: dataLoader.hasMoreRecords(),
      },
      { label: 'Leads Tracked', value: '12' },
      { label: 'Processing Rate', value: '500 Hz' },
      { label: 'Model Focus', value: 'MLP + SHAP' },
    ],
    [dataLoader, patients.length],
  );

  const scrollToDashboard = () => {
    if (!selectedPatient) {
      setShowPatientSelector(true);
      return;
    }

    scrollToMonitoring(220);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[linear-gradient(135deg,#f7fbff_0%,#edf7f2_45%,#fef7ef_100%)] px-6">
        <div className="mx-auto flex min-h-screen max-w-6xl items-center justify-center">
          <div className="w-full max-w-xl rounded-[2rem] border border-white/60 bg-white/78 p-10 text-center shadow-[0_32px_120px_rgba(15,23,42,0.12)] backdrop-blur-2xl">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
              className="mx-auto mb-5 h-16 w-16 rounded-full border-4 border-teal-500 border-t-transparent"
            />
            <h2 className="text-3xl font-semibold text-slate-900">Loading PTB-XL Database</h2>
            <p className="mt-3 text-slate-600">Preparing the first real ECG batch so the dashboard opens fast.</p>
            <div className="mt-6 h-2 overflow-hidden rounded-full bg-slate-200">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: '38%' }}
                transition={{ duration: 1.2, repeat: Infinity, repeatType: 'reverse' }}
                className="h-full rounded-full bg-[linear-gradient(90deg,#0f766e,#14b8a6,#38bdf8)]"
              />
            </div>
            <p className="mt-4 text-sm text-slate-500">Initial batch target: {INITIAL_BATCH_SIZE} patients</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[linear-gradient(180deg,#f7fbff_0%,#f0f7f2_28%,#fffaf3_100%)] text-slate-900">
      <div className="scroll-progress" />
      <div className="pointer-events-none fixed inset-x-0 top-[-10rem] z-0 mx-auto h-[32rem] w-[32rem] rounded-full bg-[radial-gradient(circle,rgba(20,184,166,0.14)_0%,rgba(56,189,248,0.08)_38%,transparent_70%)] blur-3xl" />
      <div className="pointer-events-none fixed right-[-8rem] top-[14rem] z-0 h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle,rgba(245,158,11,0.12)_0%,rgba(248,113,113,0.06)_38%,transparent_72%)] blur-3xl" />

      <header className="sticky top-0 z-50 border-b border-white/55 bg-white/70 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-[1680px] items-center justify-between gap-6 px-4 py-4 sm:px-6 lg:px-8">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.34em] text-teal-700/80">Cardiac Intelligence</p>
            <h1 className="mt-1 text-xl font-semibold text-slate-900 md:text-2xl">
              Medical Grade ECG Analyzer
            </h1>
            <p className="text-sm text-slate-600">
              PTB-XL database, 12-lead monitoring, explainable model review
            </p>
          </div>

          <div className="flex items-center gap-3">
            {selectedPatient && (
              <motion.div
                initial={{ opacity: 0, x: 18 }}
                animate={{ opacity: 1, x: 0 }}
                className="hidden rounded-2xl border border-teal-200/70 bg-white/80 px-4 py-2 text-sm text-slate-700 shadow-sm md:block"
              >
                <p className="font-medium text-slate-900">{selectedPatient.patientId}</p>
                <p className="text-xs text-slate-500">
                  {selectedPatient.age}y {selectedPatient.sex}
                </p>
              </motion.div>
            )}

            <button
              onClick={() => setShowPatientSelector(true)}
              className="rounded-full bg-[linear-gradient(135deg,#0f766e,#0891b2)] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_14px_34px_rgba(8,145,178,0.28)] transition duration-300 hover:-translate-y-0.5 hover:shadow-[0_18px_40px_rgba(8,145,178,0.34)]"
            >
              Select Patient
            </button>

            <div
              className={`rounded-full px-4 py-2 text-sm font-semibold ${
                isRecording
                  ? 'bg-rose-100 text-rose-700 ring-1 ring-rose-200'
                  : 'bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200'
              }`}
            >
              {isRecording ? 'Live Recording' : 'Standby'}
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10">
        <section className="mx-auto w-full max-w-[1680px] px-4 pb-10 pt-8 sm:px-6 md:pb-14 md:pt-12 lg:px-8" ref={heroRef}>
          <div className="grid items-center gap-8 xl:grid-cols-[1.12fr_0.88fr] xl:gap-10">
            <motion.div
              initial={{ opacity: 0, y: 36 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="inline-flex items-center gap-2 rounded-full border border-teal-200/70 bg-white/80 px-4 py-2 text-sm text-teal-800 shadow-sm">
                <span className="h-2.5 w-2.5 rounded-full bg-teal-500 shadow-[0_0_16px_rgba(20,184,166,0.75)]" />
                Clinical workflow with patient playback, signal processing, and interpretation
              </div>

              <h2 className="mt-6 max-w-4xl font-serif text-5xl leading-[1.02] text-slate-950 md:text-6xl xl:text-[5.4rem]">
                Advanced ECG analysis for research, demonstration, and expert review.
              </h2>

              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">
                Access 12-lead monitoring, denoising stages, heatmap localization, and model attribution in a
                single interface built to communicate results clearly and professionally.
              </p>

              <div className="mt-8 flex flex-wrap items-center gap-4">
                <button
                  onClick={scrollToDashboard}
                  className="rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-[0_16px_40px_rgba(15,23,42,0.24)] transition duration-300 hover:-translate-y-0.5"
                >
                  Explore Dashboard
                </button>
                <button
                  onClick={() => setShowPatientSelector(true)}
                  className="rounded-full border border-slate-300 bg-white/70 px-6 py-3 text-sm font-semibold text-slate-700 transition duration-300 hover:border-teal-300 hover:text-teal-800"
                >
                  Load a Patient
                </button>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 42 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
              className="relative"
            >
              <div className="absolute -inset-6 rounded-[2rem] bg-[conic-gradient(from_180deg_at_50%_50%,rgba(20,184,166,0.22),rgba(14,165,233,0.06),rgba(245,158,11,0.18),rgba(20,184,166,0.22))] blur-2xl" />
              <div className="relative overflow-hidden rounded-[2rem] border border-white/60 bg-slate-950 p-6 text-white shadow-[0_36px_100px_rgba(15,23,42,0.32)] xl:p-7">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(45,212,191,0.22),transparent_32%),linear-gradient(135deg,rgba(15,23,42,0.96),rgba(2,132,199,0.88))]" />
                <div className="relative">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.32em] text-teal-200/75">System Snapshot</p>
                      <h3 className="mt-2 text-2xl font-semibold">Clinical signal review</h3>
                    </div>
                    <div className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs text-teal-100">
                      {isLoadingMore
                        ? `Syncing ${loadingProgress.loaded}/${loadingProgress.total}`
                        : 'Dataset ready'}
                    </div>
                  </div>

                  <div className="mt-8 grid grid-cols-2 gap-4">
                    {stats.map((stat) => (
                      <div
                        key={stat.label}
                        className="rounded-2xl border border-white/10 bg-white/8 p-4 backdrop-blur-sm"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <p className="text-xs uppercase tracking-[0.18em] text-slate-300">{stat.label}</p>
                          {'canLoadMore' in stat && stat.canLoadMore && (
                            <button
                              type="button"
                              onClick={loadMorePatients}
                              disabled={isLoadingMore}
                              className="flex h-8 w-8 items-center justify-center rounded-full border border-white/15 bg-white/10 text-lg font-semibold text-white transition hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-50"
                              aria-label="Load 100 more patients"
                              title="Load 100 more patients"
                            >
                              {isLoadingMore ? '…' : '+'}
                            </button>
                          )}
                        </div>
                        <p className="mt-3 text-3xl font-semibold text-white">{stat.value}</p>
                        {'canLoadMore' in stat && (
                          <p className="mt-2 text-xs text-slate-300/90">
                            {stat.canLoadMore
                              ? `Tap + to load ${LOAD_MORE_BATCH_SIZE} more patients`
                              : 'All available patient records are loaded'}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="mt-8 space-y-3">
                    {sectionCards.map((card, index) => (
                      <motion.div
                        key={card.title}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.25 + index * 0.08 }}
                        className="rounded-2xl border border-white/10 bg-white/7 p-4"
                      >
                        <div className="flex items-start gap-3">
                          <div className="mt-1 h-2.5 w-2.5 rounded-full bg-teal-300" />
                          <div>
                            <p className="font-medium text-white">{card.title}</p>
                            <p className="mt-1 text-sm leading-6 text-slate-300">{card.description}</p>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="mx-auto w-full max-w-[1680px] space-y-8 px-4 pb-20 sm:px-6 lg:px-8">
          <SectionShell
            eyebrow="Monitoring"
            title="ECG monitoring workspace"
            description="Open a patient and move directly into the live waveform workspace with a broader, more focused review layout."
          >
            <section ref={monitoringSectionRef}>
              <ECGDashboard />
            </section>
          </SectionShell>

          <SectionShell
            eyebrow="Processing"
            title="Signal processing pipeline"
            description="Inspect each filtering stage in sequence inside a layout optimized for technical review and side-by-side reading."
          >
            <SignalProcessingViewer />
          </SectionShell>

          <SectionShell
            eyebrow="Interpretation"
            title="Explainability and expert review"
            description="Top leads, feature attribution, and heatmap-backed evidence are organized into a cleaner clinical review surface."
          >
            <div className="space-y-8">
              <ExpertDashboard />
              <ECGHeatmapViewer />
              <ClinicalFeaturesTable />
              <SHAPExplainability />
              <LeadWiseSHAP />
            </div>
          </SectionShell>
        </section>
      </main>

      <AnimatePresence>
        {showPatientSelector && (
          <PatientSelector
            patients={patients}
            selectedPatient={selectedPatient}
            onSelectPatient={handleSelectPatient}
            onClose={() => setShowPatientSelector(false)}
          />
        )}
      </AnimatePresence>

      <footer className="relative z-10 border-t border-white/60 bg-white/72 backdrop-blur-2xl">
        <div className="mx-auto flex w-full max-w-[1680px] flex-col gap-2 px-4 py-6 text-sm text-slate-600 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
          <p>For research and educational use only. This interface is not intended for clinical diagnosis.</p>
          <p className="text-slate-500">© 2026 Advanced ECG Analyzer with MLP Neural Network</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
