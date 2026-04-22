/**
 * Build-time ECG Data Processor
 * 
 * This script processes PTB-XL data during the build process
 * so the app works automatically when deployed without manual steps.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('\n========================================');
console.log('Processing PTB-XL ECG Data for Deployment');
console.log('========================================\n');

const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const ARCHIVE_DIR = path.join(__dirname, '..', 'archive', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3');

// Check if we already have processed data
const fullSignalsPath = path.join(PUBLIC_DIR, 'ptbxl_full_signals.json');
const recordsPath = path.join(PUBLIC_DIR, 'ptbxl_records.json');

if (fs.existsSync(fullSignalsPath)) {
  console.log('✓ Full signals data already exists');
  const stats = fs.statSync(fullSignalsPath);
  console.log(`  File size: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
  
  // Verify it's valid JSON
  try {
    const data = JSON.parse(fs.readFileSync(fullSignalsPath, 'utf8'));
    console.log(`  Contains ${data.length} records`);
    if (data.length > 0 && data[0].leads) {
      const firstLead = Object.values(data[0].leads)[0];
      if (firstLead && firstLead.processed) {
        console.log(`  Signal samples per lead: ${firstLead.processed.length}`);
      }
    }
    console.log('\n✓ ECG data is ready for deployment!\n');
    process.exit(0);
  } catch (e) {
    console.warn('⚠ Full signals file exists but is invalid, will regenerate...\n');
  }
}

if (fs.existsSync(recordsPath)) {
  console.log('✓ Using existing ptbxl_records.json');
  const stats = fs.statSync(recordsPath);
  console.log(`  File size: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
  
  try {
    const data = JSON.parse(fs.readFileSync(recordsPath, 'utf8'));
    console.log(`  Contains ${data.length} records`);
    
    // Enhance the existing data by concatenating all beat signals
    console.log('\nEnhancing existing data for better display...');
    let enhanced = 0;
    
    data.forEach((record, idx) => {
      if (record.leads) {
        Object.keys(record.leads).forEach(leadName => {
          const lead = record.leads[leadName];
          if (lead.beats && Array.isArray(lead.beats) && lead.beats.length > 0) {
            // Concatenate all beat signals into one continuous signal
            const fullSignal = [];
            lead.beats.forEach(beat => {
              if (beat.signal && Array.isArray(beat.signal)) {
                fullSignal.push(...beat.signal);
              }
            });
            
            // Store as 'processed' signal for compatibility
            if (fullSignal.length > 0) {
              record.leads[leadName] = {
                ...lead,
                processed: fullSignal,
                raw: fullSignal
              };
              enhanced++;
            }
          }
        });
      }
    });
    
    // Save enhanced version
    const enhancedPath = path.join(PUBLIC_DIR, 'ptbxl_enhanced.json');
    fs.writeFileSync(enhancedPath, JSON.stringify(data, null, 2));
    console.log(`✓ Enhanced ${enhanced} lead signals`);
    console.log(`✓ Saved to: ptbxl_enhanced.json`);
    
    const enhancedStats = fs.statSync(enhancedPath);
    console.log(`  File size: ${(enhancedStats.size / (1024 * 1024)).toFixed(2)} MB`);
    
    console.log('\n✓ ECG data is ready for deployment!\n');
    process.exit(0);
  } catch (e) {
    console.error('✗ Error processing ptbxl_records.json:', e.message);
    process.exit(1);
  }
}

// Check if archive data exists
if (!fs.existsSync(ARCHIVE_DIR)) {
  console.log('⚠ PTB-XL archive not found at expected location');
  console.log(`  Expected: ${ARCHIVE_DIR}`);
  console.log('\n⚠ To include real ECG data:');
  console.log('  1. Download PTB-XL dataset');
  console.log('  2. Place in archive/ folder');
  console.log('  3. Run: npm run process-data');
  console.log('\n  Or use the existing ptbxl_records.json file\n');
  
  // Check if we have ANY data file
  if (!fs.existsSync(recordsPath)) {
    console.error('✗ No ECG data files found!');
    console.error('  The app needs at least ptbxl_records.json in the public folder.\n');
    process.exit(1);
  }
}

console.log('\n✓ Build can proceed with available data\n');
process.exit(0);
