import json
from collections import Counter

records = json.load(open('public/ptbxl_records.json'))
print(f'Total records: {len(records)}')
print()

for i, r in enumerate(records[:30]):
    pid = r.get('patient_id', r.get('ecg_id'))
    diag = r.get('diagnosis', 'N/A')
    print(f'  {pid}: {diag}')

print()
diags = [r.get('diagnosis', 'N/A') for r in records]
counts = Counter(diags)
print('Top 30 diagnoses:')
for d, c in counts.most_common(30):
    print(f'  {c:4d} - {d}')

# Check the raw SCP codes
print()
print('Sample SCP codes:')
for r in records[:5]:
    scp = r.get('scp_codes', {})
    pid = r.get('patient_id', r.get('ecg_id'))
    print(f'  {pid}: {scp}')
