import React from 'react';
import { motion } from 'framer-motion';
import { useECGStore } from '../store/ecgStore';

const ControlPanel: React.FC = () => {
  const {
    speed,
    amplitude,
    isAnimating,
    setSpeed,
    setAmplitude,
    setAnimating,
    setRecording
  } = useECGStore();

  console.log('ControlPanel render - isAnimating:', isAnimating);

  const handleToggleAnimation = () => {
    const newState = !isAnimating;
    setAnimating(newState);
    setRecording(newState); // Control recording along with animation
  };

  return (
    <div className="space-y-6">
      {/* Display Settings */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8 border border-gray-200 hover:shadow-2xl transition-shadow duration-300"
      >
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
          </div>
          <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            Display Settings
          </h3>
        </div>
        
        <div className="space-y-6">
          {/* Speed Control */}
          <div className="relative">
            <div className="flex justify-between mb-3">
              <label className="text-sm font-semibold text-gray-700 flex items-center space-x-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Scroll Speed</span>
              </label>
              <motion.span 
                key={speed}
                initial={{ scale: 1.2 }}
                animate={{ scale: 1 }}
                className="text-sm font-bold text-blue-600 bg-blue-50 px-3 py-1 rounded-full"
              >
                {speed}x
              </motion.span>
            </div>
            <div className="relative">
              <input
                type="range"
                min="0.5"
                max="3"
                step="0.5"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full h-3 bg-gradient-to-r from-blue-100 to-blue-200 rounded-full appearance-none cursor-pointer slider-thumb-blue"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((speed - 0.5) / 2.5) * 100}%, #dbeafe ${((speed - 0.5) / 2.5) * 100}%, #dbeafe 100%)`
                }}
              />
            </div>
            <div className="flex justify-between text-xs font-medium text-gray-500 mt-2">
              <span className="bg-gray-100 px-2 py-1 rounded">0.5x</span>
              <span className="bg-gray-100 px-2 py-1 rounded">3x</span>
            </div>
          </div>

          {/* Amplitude Control */}
          <div className="relative">
            <div className="flex justify-between mb-3">
              <label className="text-sm font-semibold text-gray-700 flex items-center space-x-2">
                <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                <span>Amplitude</span>
              </label>
              <motion.span 
                key={amplitude}
                initial={{ scale: 1.2 }}
                animate={{ scale: 1 }}
                className="text-sm font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full"
              >
                {amplitude}x
              </motion.span>
            </div>
            <div className="relative">
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={amplitude}
                onChange={(e) => setAmplitude(parseFloat(e.target.value))}
                className="w-full h-3 bg-gradient-to-r from-emerald-100 to-emerald-200 rounded-full appearance-none cursor-pointer slider-thumb-emerald"
                style={{
                  background: `linear-gradient(to right, #10b981 0%, #10b981 ${((amplitude - 0.5) / 1.5) * 100}%, #d1fae5 ${((amplitude - 0.5) / 1.5) * 100}%, #d1fae5 100%)`
                }}
              />
            </div>
            <div className="flex justify-between text-xs font-medium text-gray-500 mt-2">
              <span className="bg-gray-100 px-2 py-1 rounded">0.5x</span>
              <span className="bg-gray-100 px-2 py-1 rounded">2x</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Animation Control */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-8 border border-gray-200 hover:shadow-2xl transition-shadow duration-300"
      >
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-pink-600">
            Animation Control
          </h3>
        </div>
        
        <div className="space-y-4">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleToggleAnimation}
            className={`w-full py-4 px-6 rounded-xl font-bold text-white shadow-lg transition-all duration-300 flex items-center justify-center space-x-3 ${
              isAnimating 
                ? 'bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700' 
                : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700'
            }`}
          >
            {isAnimating ? (
              <>
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Pause ECG Motion</span>
              </>
            ) : (
              <>
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Resume ECG Motion</span>
              </>
            )}
          </motion.button>
          
          <div className={`text-center text-sm font-semibold p-3 rounded-lg ${
            isAnimating 
              ? 'bg-orange-50 text-orange-700 border border-orange-200' 
              : 'bg-gray-100 text-gray-600 border border-gray-200'
          }`}>
            {isAnimating ? 'ECG graphs are scrolling' : 'ECG graphs are paused'}
          </div>
        </div>
      </motion.div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl p-6 border border-gray-200 hover:shadow-2xl transition-shadow duration-300"
      >
        <div className="space-y-4">
          <motion.div 
            whileHover={{ x: 5 }}
            className="flex items-center space-x-3 p-3 bg-green-50 rounded-xl border border-green-200"
          >
            <div className="relative">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <div className="absolute inset-0 w-3 h-3 bg-green-400 rounded-full animate-ping"></div>
            </div>
            <div className="flex-1">
              <span className="text-sm font-bold text-green-700">System Online</span>
              <div className="w-full bg-green-200 h-1 rounded-full mt-1">
                <div className="bg-green-500 h-1 rounded-full w-full"></div>
              </div>
            </div>
            <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </motion.div>

          <motion.div 
            whileHover={{ x: 5 }}
            className="flex items-center space-x-3 p-3 bg-blue-50 rounded-xl border border-blue-200"
          >
            <div className="relative">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <div className="absolute inset-0 w-3 h-3 bg-blue-400 rounded-full animate-ping"></div>
            </div>
            <div className="flex-1">
              <span className="text-sm font-bold text-blue-700">MLP Model Active</span>
              <div className="w-full bg-blue-200 h-1 rounded-full mt-1">
                <div className="bg-blue-500 h-1 rounded-full w-full"></div>
              </div>
            </div>
            <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M13 7H7v6h6V7z" />
              <path fillRule="evenodd" d="M7 2a1 1 0 012 0v1h2V2a1 1 0 112 0v1h2a2 2 0 012 2v2h1a1 1 0 110 2h-1v2h1a1 1 0 110 2h-1v2a2 2 0 01-2 2h-2v1a1 1 0 11-2 0v-1H9v1a1 1 0 11-2 0v-1H5a2 2 0 01-2-2v-2H2a1 1 0 110-2h1V9H2a1 1 0 010-2h1V5a2 2 0 012-2h2V2zM5 5h10v10H5V5z" clipRule="evenodd" />
            </svg>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default ControlPanel;
