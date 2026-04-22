import pandas as pd
import ast

df = pd.read_csv('public/ptbxl_database.csv')
print('Shape:', df.shape)
print('Columns:', list(df.columns[:20]))
print()

# Parse SCP codes and count class distributions
print('sample scp_codes:')
for i, row in df.head(10).iterrows():
    print(f'  ecg_id={row["ecg_id"]}: {row["scp_codes"]}')

if 'diagnostic_class' in df.columns:
    print('\ndiagnostic_class distribution:')
    print(df['diagnostic_class'].value_counts().to_string())

if 'diagnostic_superclass' in df.columns:
    print('\ndiagnostic_superclass distribution:')
    print(df['diagnostic_superclass'].value_counts().head(30).to_string())

# Count the actual superclass distribution from scp codes
print('\nParsing SCP codes for superclass distribution...')
scp_df = pd.read_csv('archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv')
print('SCP statements shape:', scp_df.shape)
print('SCP columns:', list(scp_df.columns))
print()
print('Unique diagnostic_class in scp_statements:')
if 'diagnostic_class' in scp_df.columns:
    print(scp_df[['Statement Description', 'diagnostic_class', 'diagnostic_subclass']].dropna(subset=['diagnostic_class']).head(30).to_string())
