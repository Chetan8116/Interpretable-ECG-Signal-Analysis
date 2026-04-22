/**
 * Detailed test to understand CSV structure
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Read the CSV file
const csvPath = path.join(__dirname, 'public', 'stpetersburg_segment_features_selected.csv');
const csvContent = fs.readFileSync(csvPath, 'utf-8');
const lines = csvContent.trim().split('\n');
const headers = lines[0].split(',');

console.log('=== Detailed CSV Structure Analysis ===\n');

// Get unique patient combinations
const patientCombos = new Set();
const leadCounts = {};

for (let i = 1; i < Math.min(5000, lines.length); i++) {
  const values = lines[i].split(',');
  const age = values[0];
  const sex = values[1];
  const leadIdx = values[2];
  
  const patientKey = `${age}-${sex}`;
  patientCombos.add(patientKey);
  
  if (!leadCounts[patientKey]) {
    leadCounts[patientKey] = {};
  }
  if (!leadCounts[patientKey][leadIdx]) {
    leadCounts[patientKey][leadIdx] = 0;
  }
  leadCounts[patientKey][leadIdx]++;
}

console.log(`Unique patient combinations in first 5000 rows: ${patientCombos.size}`);
for (const combo of Array.from(patientCombos).slice(0, 5)) {
  const leads = leadCounts[combo];
  const totalRecords = Object.values(leads).reduce((a, b) => a + b, 0);
  const uniqueLeads = Object.keys(leads).length;
  const beatsPerLead = Object.values(leads)[0];
  
  console.log(`\n  Patient ${combo}:`);
  console.log(`    Total records: ${totalRecords}`);
  console.log(`    Unique leads: ${uniqueLeads}`);
  console.log(`    Records per lead: ${beatsPerLead}`);
  console.log(`    Beats: ${Math.floor(totalRecords / 12)} (if all 12 leads)`);
}

console.log('\n=== CSV Interpretation ===');
console.log('The CSV appears to contain ECG feature data where:');
console.log('- Each row represents 1 lead of 1 beat for 1 patient');
console.log('- Patients are identified by (age, sex) combination');
console.log('- Each patient has multiple beats (cardiac cycles)');
console.log('- Each beat has 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)');
console.log('- Total records = (# patients) × (# beats per patient) × 12');
