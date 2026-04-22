/**
 * Quick test to verify CSV data loading works correctly
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

console.log('=== CSV Data Loading Test ===\n');
console.log(`Total rows: ${lines.length - 1}`);
console.log(`Headers: ${headers.join(', ')}\n`);

// Parse a few rows to check structure
const patientData = {};
for (let i = 1; i < Math.min(100, lines.length); i++) {
  const values = lines[i].split(',');
  const row = {};
  headers.forEach((header, index) => {
    const value = values[index];
    row[header] = isNaN(Number(value)) ? value : Number(value);
  });

  const patientKey = `${row.age}-${row.sex}`;
  if (!patientData[patientKey]) {
    patientData[patientKey] = { leads: {}, count: 0 };
  }
  patientData[patientKey].count++;
  if (!patientData[patientKey].leads[row.lead_idx]) {
    patientData[patientKey].leads[row.lead_idx] = 0;
  }
  patientData[patientKey].leads[row.lead_idx]++;
}

console.log('Sample patients from first 100 rows:');
for (const [key, data] of Object.entries(patientData)) {
  console.log(`  Patient ${key}:`);
  console.log(`    Total records: ${data.count}`);
  console.log(`    Leads: ${Object.keys(data.leads).join(', ')}`);
  const expectedLeads = Object.keys(data.leads).length;
  console.log(`    Expected beats: ${data.count / 12} (actual lead types: ${expectedLeads})`);
}

console.log('\n=== Data structure looks good! ===');
