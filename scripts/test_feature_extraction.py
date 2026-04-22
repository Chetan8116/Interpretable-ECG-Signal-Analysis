"""
Quick test script for comprehensive feature extraction
Tests on a single PTB-XL record to verify all features work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from extract_comprehensive_features_ptbxl import *

def test_single_record():
    """Test feature extraction on a single record"""
    
    print("=" * 80)
    print("TESTING COMPREHENSIVE FEATURE EXTRACTION")
    print("=" * 80)
    print()
    
    # Test configuration
    test_record_path = "archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/00000/00001_hr"
    test_record_id = 1
    
    if not os.path.exists(test_record_path + ".hea"):
        print(f"ERROR: Test record not found at {test_record_path}")
        print("Please update the path in this script.")
        return False
    
    print(f"Test Record: {test_record_path}")
    print(f"Record ID: {test_record_id}")
    print()
    
    # Process the record
    result = process_record(test_record_path, test_record_id, SAMPLING_FREQ)
    
    if not result['success']:
        print(f"\nERROR: Feature extraction failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False
    
    # Print results
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION SUCCESSFUL")
    print("=" * 80)
    print()
    
    print(f"Record ID: {result['record_id']}")
    print(f"Number of leads processed: {len(result['features'])}")
    print()
    
    # Analyze features from first lead
    first_lead_key = list(result['features'].keys())[0]
    first_lead_features = result['features'][first_lead_key]
    
    print(f"Sample Features from {first_lead_key}:")
    print("-" * 80)
    
    # Count features by category
    categories = {
        'stat': [],
        'temp': [],
        'morph': [],
        'spec': [],
        'tf': [],
        'nonlin': []
    }
    
    for feature_name, feature_value in first_lead_features.items():
        if feature_name in ['lead_name', 'sampling_freq']:
            continue
        
        category = feature_name.split('_')[0]
        if category in categories:
            categories[category].append((feature_name, feature_value))
    
    # Print feature counts and samples
    total_features = 0
    for cat_name, features in categories.items():
        print(f"\n{cat_name.upper()} Features ({len(features)}):")
        for i, (fname, fval) in enumerate(features[:5]):  # Show first 5 of each category
            if isinstance(fval, float):
                print(f"  {fname}: {fval:.6f}")
            else:
                print(f"  {fname}: {fval}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")
        total_features += len(features)
    
    print()
    print("=" * 80)
    print(f"Total Features Extracted per Lead: {total_features}")
    print(f"Total Features for All {len(result['features'])} Leads: {total_features * len(result['features'])}")
    print("=" * 80)
    print()
    
    # Check for NaN values
    nan_count = sum(1 for fname, fval in first_lead_features.items() 
                    if isinstance(fval, float) and np.isnan(fval))
    print(f"Features with NaN values in {first_lead_key}: {nan_count}")
    
    if nan_count > 0:
        print("Note: Some NaN values are expected for certain features depending on signal quality")
    
    print()
    print("✓ Feature extraction test PASSED!")
    print()
    
    # Save test results
    test_output_dir = Path("test_features")
    test_output_dir.mkdir(exist_ok=True)
    
    import json
    test_file = test_output_dir / "test_single_record_features.json"
    with open(test_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Test results saved to: {test_file}")
    print()
    
    return True


def test_feature_functions():
    """Test individual feature extraction functions"""
    
    print("=" * 80)
    print("TESTING INDIVIDUAL FEATURE FUNCTIONS")
    print("=" * 80)
    print()
    
    # Create synthetic ECG signal
    fs = 500
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Simulated ECG with QRS complexes
    # Simple sine wave with spikes to simulate R-peaks
    ecg_signal = 0.1 * np.sin(2 * np.pi * 1.2 * t)  # Baseline
    
    # Add R-peaks (simulated at ~60 bpm = 1 Hz)
    for i in range(int(duration)):
        peak_idx = int((i + 0.5) * fs)
        if peak_idx < len(ecg_signal):
            ecg_signal[peak_idx] += 1.0  # R-peak
            # Add some QRS shape
            for j in range(-20, 20):
                idx = peak_idx + j
                if 0 <= idx < len(ecg_signal):
                    ecg_signal[idx] += 0.3 * np.exp(-(j**2) / 100)
    
    # Add noise
    ecg_signal += 0.05 * np.random.randn(len(ecg_signal))
    
    print("Testing on Synthetic ECG Signal:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling Frequency: {fs} Hz")
    print(f"  Signal Length: {len(ecg_signal)} samples")
    print()
    
    # Test each feature category
    tests = [
        ("Statistical", extract_statistical_features, (ecg_signal,)),
        ("Temporal", extract_temporal_features, (ecg_signal, fs)),
        ("Morphological", extract_morphological_features, (ecg_signal, fs)),
        ("Spectral", extract_spectral_features, (ecg_signal, fs)),
        ("Time-Frequency", extract_time_frequency_features, (ecg_signal, fs)),
        ("Nonlinear", extract_nonlinear_features, (ecg_signal, fs))
    ]
    
    all_passed = True
    
    for test_name, test_func, test_args in tests:
        try:
            print(f"Testing {test_name} features...")
            features = test_func(*test_args)
            
            valid_count = sum(1 for v in features.values() if not (isinstance(v, float) and np.isnan(v)))
            total_count = len(features)
            
            print(f"  ✓ {test_name}: {valid_count}/{total_count} features extracted successfully")
            
            # Show a few sample features
            sample_features = list(features.items())[:3]
            for fname, fval in sample_features:
                if isinstance(fval, float):
                    print(f"    - {fname}: {fval:.6f}")
                else:
                    print(f"    - {fname}: {fval}")
            print()
            
        except Exception as e:
            print(f"  ✗ {test_name}: FAILED - {str(e)}")
            traceback.print_exc()
            all_passed = False
            print()
    
    if all_passed:
        print("=" * 80)
        print("✓ All feature function tests PASSED!")
        print("=" * 80)
    else:
        print("=" * 80)
        print("✗ Some tests FAILED - check errors above")
        print("=" * 80)
    
    print()
    return all_passed


if __name__ == "__main__":
    print("\n")
    
    # Test individual functions first
    functions_ok = test_feature_functions()
    
    print("\n" + "=" * 80 + "\n")
    
    # Then test on real record
    if functions_ok:
        record_ok = test_single_record()
        
        if record_ok:
            print("\n" + "=" * 80)
            print("ALL TESTS PASSED - Ready to process full dataset!")
            print("=" * 80)
            print()
            print("To process the entire PTB-XL dataset, run:")
            print("  python scripts/extract_comprehensive_features_ptbxl.py")
            print()
        else:
            print("\nRecord test failed. Please check the paths and try again.")
    else:
        print("\nFunction tests failed. Please fix the errors before processing records.")
