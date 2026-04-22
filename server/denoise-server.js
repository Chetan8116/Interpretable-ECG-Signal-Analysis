import express from 'express';
import cors from 'cors';
import { exec, spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// Store running processes
const runningProcesses = new Map();

// Check if Python is available
async function checkPythonAvailable() {
  return new Promise((resolve) => {
    // Try 'py' first (Windows Python launcher)
    exec('py --version', (error, stdout, stderr) => {
      if (!error) {
        resolve(true);
      } else {
        // Fallback to python/python3
        exec('python --version', (error2, stdout2, stderr2) => {
          if (error2) {
            exec('python3 --version', (error3, stdout3, stderr3) => {
              resolve(!error3);
            });
          } else {
            resolve(true);
          }
        });
      }
    });
  });
}

// Get Python command (py, python, or python3)
function getPythonCommand() {
  return new Promise((resolve) => {
    // Try 'py' first (Windows Python launcher)
    exec('py --version', (error) => {
      if (!error) {
        resolve('py');
      } else {
        // Fallback to python/python3
        exec('python --version', (error2) => {
          resolve(error2 ? 'python3' : 'python');
        });
      }
    });
  });
}

// Check denoising status
app.get('/api/denoise/status', async (req, res) => {
  const projectRoot = path.join(__dirname, '..');
  const metadataPath = path.join(projectRoot, 'ptbxl_denoised', 'processed_metadata.csv');
  const partialMetadataPath = path.join(projectRoot, 'ptbxl_denoised', 'processed_metadata_partial.csv');
  const featureSummaryPath = path.join(projectRoot, 'ptbxl_denoised', 'feature_preservation_summary.json');
  
  try {
    // Check if denoising has been completed
    let completed = false;
    let processedRecords = 0;
    
    try {
      await fs.access(metadataPath);
      completed = true;
      const content = await fs.readFile(metadataPath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());
      processedRecords = Math.floor((lines.length - 1) / 12); // 12 leads per record
    } catch (err) {
      // Check partial
      try {
        await fs.access(partialMetadataPath);
        const content = await fs.readFile(partialMetadataPath, 'utf-8');
        const lines = content.split('\n').filter(line => line.trim());
        processedRecords = Math.floor((lines.length - 1) / 12);
      } catch (err2) {
        // No files yet
      }
    }
    
    const isRunning = runningProcesses.size > 0;
    let featurePreservationSummary = null;

    try {
      await fs.access(featureSummaryPath);
      const content = await fs.readFile(featureSummaryPath, 'utf-8');
      featurePreservationSummary = JSON.parse(content);
    } catch (err) {
      // summary not available yet
    }
    
    res.json({
      completed,
      isRunning,
      processedRecords,
      pythonAvailable: await checkPythonAvailable(),
      featurePreservationSummary
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start denoising process
app.post('/api/denoise/start', async (req, res) => {
  const { patientId, targetFs = 500 } = req.body;
  
  if (!patientId) {
    return res.status(400).json({ error: 'Patient ID is required' });
  }
  
  try {
    const pythonAvailable = await checkPythonAvailable();
    if (!pythonAvailable) {
      return res.status(400).json({ 
        error: 'Python not found. Please install Python and required packages.' 
      });
    }
    
    if (runningProcesses.size > 0) {
      return res.status(400).json({ 
        error: 'Denoising process already running' 
      });
    }
    
    const projectRoot = path.join(__dirname, '..');
    const scriptPath = path.join(projectRoot, 'scripts', 'denoise_ptbxl_data.py');
    const pythonCmd = await getPythonCommand();
    
    // Create modified script with custom parameters
    const processId = Date.now().toString();
    
    const pythonProcess = spawn(pythonCmd, [
      scriptPath,
      '--patient-id', patientId,
      '--target-fs', targetFs.toString()
    ], {
      cwd: projectRoot,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    const processInfo = {
      id: processId,
      process: pythonProcess,
      output: [],
      error: [],
      startTime: Date.now()
    };
    
    runningProcesses.set(processId, processInfo);
    
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      processInfo.output.push(output);
      console.log('[Python stdout]:', output);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      processInfo.error.push(error);
      console.log('[Python stderr]:', error);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`[Python] Process exited with code ${code}`);
      setTimeout(() => runningProcesses.delete(processId), 60000); // Keep for 1 min
    });
    
    res.json({ 
      success: true, 
      processId,
      message: `Started denoising patient ${patientId} at ${targetFs}Hz` 
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get process output
app.get('/api/denoise/output/:processId', (req, res) => {
  const { processId } = req.params;
  const processInfo = runningProcesses.get(processId);
  
  if (!processInfo) {
    return res.status(404).json({ error: 'Process not found' });
  }
  
  res.json({
    output: processInfo.output.join(''),
    error: processInfo.error.join(''),
    running: processInfo.process.exitCode === null,
    exitCode: processInfo.process.exitCode
  });
});

// Stop denoising process
app.post('/api/denoise/stop/:processId', (req, res) => {
  const { processId } = req.params;
  const processInfo = runningProcesses.get(processId);
  
  if (!processInfo) {
    return res.status(404).json({ error: 'Process not found' });
  }
  
  processInfo.process.kill();
  runningProcesses.delete(processId);
  
  res.json({ success: true, message: 'Process stopped' });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Feature extraction endpoints
app.post('/api/features/extract', async (req, res) => {
  const { patientId } = req.body;
  
  if (!patientId) {
    return res.status(400).json({ error: 'Patient ID is required' });
  }
  
  try {
    const pythonAvailable = await checkPythonAvailable();
    if (!pythonAvailable) {
      return res.status(400).json({ 
        error: 'Python not found. Please install Python and required packages.' 
      });
    }
    
    if (runningProcesses.size > 5) {
      return res.status(400).json({ 
        error: 'Too many processes running. Please wait.' 
      });
    }
    
    const projectRoot = path.join(__dirname, '..');
    const scriptPath = path.join(projectRoot, 'scripts', 'feature_enhancement.py');
    const pythonCmd = await getPythonCommand();
    
    const processId = `feat_${Date.now()}`;
    
    const pythonProcess = spawn(pythonCmd, [
      scriptPath,
      '--patient-id', patientId,
      '--denoised-dir', 'ptbxl_denoised',
      '--output-dir', 'ptbxl_features',
      '--fs', '500'
    ], {
      cwd: projectRoot,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    const processInfo = {
      id: processId,
      process: pythonProcess,
      output: [],
      error: [],
      startTime: Date.now(),
      type: 'feature_extraction'
    };
    
    runningProcesses.set(processId, processInfo);
    
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      processInfo.output.push(output);
      console.log('[Python Feature stdout]:', output);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      processInfo.error.push(error);
      console.log('[Python Feature stderr]:', error);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`[Python Feature] Process exited with code ${code}`);
      setTimeout(() => runningProcesses.delete(processId), 60000);
    });
    
    res.json({ 
      success: true, 
      processId,
      message: `Started feature extraction for patient ${patientId}` 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/features/status/:processId', (req, res) => {
  const { processId } = req.params;
  const processInfo = runningProcesses.get(processId);
  
  if (!processInfo) {
    return res.status(404).json({ error: 'Process not found' });
  }
  
  res.json({
    output: processInfo.output.join(''),
    error: processInfo.error.join(''),
    running: processInfo.process.exitCode === null,
    exitCode: processInfo.process.exitCode,
    type: processInfo.type
  });
});

app.get('/api/features/results/:patientId', async (req, res) => {
  const { patientId } = req.params;
  const projectRoot = path.join(__dirname, '..');
  const summaryPath = path.join(projectRoot, 'ptbxl_features', patientId, 'summary.json');
  
  try {
    const summaryData = await fs.readFile(summaryPath, 'utf-8');
    const summary = JSON.parse(summaryData);
    res.json({ success: true, patientId, features: summary });
  } catch (error) {
    res.status(404).json({ 
      error: `Features not found for patient ${patientId}. Has extraction completed?` 
    });
  }
});

// Prediction endpoint
app.post('/api/predict', async (req, res) => {
  const { patientId } = req.body;
  
  if (!patientId) {
    return res.status(400).json({ error: 'Patient ID is required' });
  }
  
  try {
    const pythonAvailable = await checkPythonAvailable();
    if (!pythonAvailable) {
      return res.status(400).json({ 
        error: 'Python not found. Please install Python and required packages.' 
      });
    }
    
    const projectRoot = path.join(__dirname, '..');
    const scriptPath = path.join(projectRoot, 'scripts', 'predict_diagnosis.py');
    const pythonCmd = await getPythonCommand();
    
    // Run prediction synchronously
    exec(`${pythonCmd} "${scriptPath}" ${patientId}`, {
      cwd: projectRoot,
      maxBuffer: 1024 * 1024 * 10 // 10MB buffer
    }, (error, stdout, stderr) => {
      if (error) {
        console.error('[Prediction Error]:', error);
        console.error('[Prediction Stderr]:', stderr);
        return res.status(500).json({ 
          error: 'Prediction failed',
          details: stderr || error.message 
        });
      }
      
      try {
        const result = JSON.parse(stdout);
        res.json(result);
      } catch (parseError) {
        console.error('[JSON Parse Error]:', parseError);
        console.error('[Raw stdout]:', stdout);
        res.status(500).json({ 
          error: 'Failed to parse prediction result',
          details: stdout 
        });
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Denoising API server running on http://localhost:${PORT}`);
  console.log(`📊 Status endpoint: http://localhost:${PORT}/api/denoise/status`);
  console.log(`🧠 Prediction endpoint: http://localhost:${PORT}/api/predict`);
});
