/**
 * Optimize ECG Data for Production Deployment
 * 
 * This script creates an optimized version of the ECG data
 * by combining beat segments into continuous signals.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const INPUT_FILE = path.join(PUBLIC_DIR, 'ptbxl_records.json');
const OUTPUT_FILE = path.join(PUBLIC_DIR, 'ptbxl_optimized.json');

console.log('\n========================================');
console.log('Optimizing ECG Data for Production');
console.log('========================================\n');

if (!fs.existsSync(INPUT_FILE)) {
  console.error('✗ ptbxl_records.json not found in public folder!');
  console.error('  Cannot optimize data without source file.\n');
  process.exit(1);
}

// Main async function
(async () => {
try {
  console.log('Loading ptbxl_records.json...');
  const rawData = fs.readFileSync(INPUT_FILE, 'utf8');
  const data = JSON.parse(rawData);
  
  console.log(`✓ Loaded ${data.length} records\n`);
  
  console.log('Processing records...');
  let totalLeadsProcessed = 0;
  let totalSamplesGenerated = 0;
  
  const optimizedData = data.map((record, idx) => {
    const optimizedRecord = {
      ecg_id: record.ecg_id,
      patient_id: record.patient_id,
      age: record.age,
      sex: record.sex,
      sampling_rate: 500, // Use 500Hz for high-quality ECG
      duration: record.duration || 10.0,
      diagnosis: record.diagnosis || '',
      scp_codes: record.scp_codes || '{}',
      leads: {}
    };
    
    if (record.leads) {
      Object.keys(record.leads).forEach(leadName => {
        const lead = record.leads[leadName];
        
        // Strategy 1: Use existing processed/raw signal if available
        if (lead.processed && Array.isArray(lead.processed) && lead.processed.length > 0) {
          optimizedRecord.leads[leadName] = {
            processed: lead.processed,
            raw: lead.raw || lead.processed
          };
          totalSamplesGenerated += lead.processed.length;
          totalLeadsProcessed++;
          return;
        }
        
        // Strategy 2: Concatenate all beat signals
        if (lead.beats && Array.isArray(lead.beats) && lead.beats.length > 0) {
          const continuousSignal = [];
          
          lead.beats.forEach(beat => {
            if (beat.signal && Array.isArray(beat.signal)) {
              continuousSignal.push(...beat.signal);
            }
          });
          
          if (continuousSignal.length > 0) {
            // Repeat signal to get approximately 10 seconds of data
            const targetSamples = optimizedRecord.sampling_rate * 10; // 10 seconds
            const repeatedSignal = [];
            
            while (repeatedSignal.length < targetSamples) {
              repeatedSignal.push(...continuousSignal);
            }
            
            // Trim to exact target length
            const finalSignal = repeatedSignal.slice(0, targetSamples);
            
            optimizedRecord.leads[leadName] = {
              processed: finalSignal,
              raw: finalSignal
            };
            totalSamplesGenerated += finalSignal.length;
            totalLeadsProcessed++;
          }
        }
        
        // Strategy 3: Use signal_preview if available
        else if (lead.signal_preview && Array.isArray(lead.signal_preview)) {
          const targetSamples = optimizedRecord.sampling_rate * 10;
          const repeatedSignal = [];
          
          while (repeatedSignal.length < targetSamples) {
            repeatedSignal.push(...lead.signal_preview);
          }
          
          const finalSignal = repeatedSignal.slice(0, targetSamples);
          
          optimizedRecord.leads[leadName] = {
            processed: finalSignal,
            raw: finalSignal
          };
          totalSamplesGenerated += finalSignal.length;
          totalLeadsProcessed++;
        }
      });
    }
    
    if ((idx + 1) % 10 === 0) {
      process.stdout.write(`\r  Processed ${idx + 1}/${data.length} records...`);
    }
    
    return optimizedRecord;
  });
  
  console.log(`\r  Processed ${data.length}/${data.length} records... Done!`);
  console.log(`\n✓ Processed ${totalLeadsProcessed} leads`);
  console.log(`✓ Generated ${totalSamplesGenerated.toLocaleString()} total samples`);
  console.log(`  Average samples per lead: ${Math.round(totalSamplesGenerated / totalLeadsProcessed)}`);
  
  // Save optimized data using streaming approach to avoid memory issues
  console.log('\nSaving optimized data...');
  
  // Write pretty-printed version in chunks
  const writeStream = fs.createWriteStream(OUTPUT_FILE);
  writeStream.write('[\n');
  
  for (let i = 0; i < optimizedData.length; i++) {
    const recordJson = JSON.stringify(optimizedData[i], null, 2);
    // Indent each line of the record
    const indentedRecord = recordJson.split('\n').map(line => '  ' + line).join('\n');
    writeStream.write(indentedRecord);
    
    if (i < optimizedData.length - 1) {
      writeStream.write(',\n');
    } else {
      writeStream.write('\n');
    }
    
    if ((i + 1) % 50 === 0) {
      process.stdout.write(`\r  Saved ${i + 1}/${optimizedData.length} records...`);
    }
  }
  
  writeStream.write(']');
  writeStream.end();
  
  // Wait for stream to finish
  await new Promise((resolve, reject) => {
    writeStream.on('finish', resolve);
    writeStream.on('error', reject);
  });
  
  console.log(`\r  Saved ${optimizedData.length}/${optimizedData.length} records... Done!`);
  
  const stats = fs.statSync(OUTPUT_FILE);
  console.log(`✓ Saved to: ${path.basename(OUTPUT_FILE)}`);
  console.log(`  File size: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
  
  // Save minified version using streaming
  console.log('\nSaving minified version...');
  const minifiedPath = OUTPUT_FILE.replace('.json', '.min.json');
  const minStream = fs.createWriteStream(minifiedPath);
  minStream.write('[');
  
  for (let i = 0; i < optimizedData.length; i++) {
    minStream.write(JSON.stringify(optimizedData[i]));
    if (i < optimizedData.length - 1) {
      minStream.write(',');
    }
    
    if ((i + 1) % 50 === 0) {
      process.stdout.write(`\r  Saved ${i + 1}/${optimizedData.length} records...`);
    }
  }
  
  minStream.write(']');
  minStream.end();
  
  await new Promise((resolve, reject) => {
    minStream.on('finish', resolve);
    minStream.on('error', reject);
  });
  
  console.log(`\r  Saved ${optimizedData.length}/${optimizedData.length} records... Done!`);
  
  const minStats = fs.statSync(minifiedPath);
  console.log(`✓ Saved minified: ${path.basename(minifiedPath)}`);
  console.log(`  File size: ${(minStats.size / (1024 * 1024)).toFixed(2)} MB (${Math.round((1 - minStats.size / stats.size) * 100)}% smaller)`);
  
  console.log('\n✓ Optimization complete!\n');
  console.log('The app will now use optimized data automatically.\n');
  
  process.exit(0);
  
} catch (error) {
  console.error('\n✗ Error during optimization:', error.message);
  console.error(error.stack);
  process.exit(1);
}
})(); // End async IIFE
