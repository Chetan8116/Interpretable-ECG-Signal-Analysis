/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        medical: {
          primary: '#0066cc',
          secondary: '#00a86b',
          danger: '#dc3545',
          warning: '#ffc107',
          dark: '#1a1a2e',
          light: '#f0f4f8'
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ecg-wave': 'ecg 2s linear infinite',
      },
      keyframes: {
        ecg: {
          '0%, 100%': { transform: 'translateX(0)' },
          '50%': { transform: 'translateX(-50%)' },
        }
      }
    },
  },
  plugins: [],
}
