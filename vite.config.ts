import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    fs: {
      // Allow serving files from archive folder
      allow: ['..']
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  },
  publicDir: 'public'
})
