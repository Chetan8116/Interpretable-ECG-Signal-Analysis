import fs from 'fs';

const csv = fs.readFileSync('public/stpetersburg_segment_features_selected.csv', 'utf-8');
const lines = csv.split('\n');
const headers = lines[0].split(',');

console.log('Analyzing lead-specific features...\n');

const rows = [];
for (let i = 1; i <= 100; i++) {
  const vals = lines[i].split(',');
  const row = {};
  headers.forEach((h, idx) => row[h] = vals[idx]);
  if (row.age === '49' && row.sex === 'F') {
    rows.push(row);
  }
}

console.log(`Found ${rows.length} rows for patient 49-F in first 100 lines\n`);

// Show features for each of the 12 leads
const leadData = {};
rows.forEach(r => {
  const leadIdx = r.lead_idx;
  if (!leadData[leadIdx]) leadData[leadIdx] = [];
  leadData[leadIdx].push(r);
});

console.log('Lead-specific amplitudes and areas:\n');
for (let i = 0; i < 12; i++) {
  if (leadData[i] && leadData[i].length > 0) {
    const r = leadData[i][0];
    console.log(`Lead ${i}: R_amp=${r.R_amp}, P_amp=${r.P_amp}, T_amp=${r.T_amp}, QRS_area=${r.QRS_area}, ST_slope=${r.ST_slope}`);
  }
}
