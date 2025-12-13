#!/usr/bin/env python3
"""
Analyze sensitivity experiment results and generate summary tables/plots.
Usage: python analyze_sensitivity.py [0.5B|1.5B]
"""

import re
import sys
import os
import glob
from pathlib import Path
from collections import defaultdict
import json

def extract_epoch_results(log_file):
    """Extract final epoch validation accuracy from log file."""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
     
    epoch_pattern = r'Epoch (\d+).*?Val Acc: ([\d.]+)'
    accuracy = None
    last_epoch = -1
    
    for line in lines:
        match = re.search(epoch_pattern, line)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > last_epoch:
                last_epoch = epoch_num
                accuracy = float(match.group(2))
    
    # Also try to get test accuracy if available
    test_acc_pattern = r'Accuracy: ([\d.]+)'
    test_accuracy = None
    for line in reversed(lines):
        match = re.search(test_acc_pattern, line)
        if match:
            test_accuracy = float(match.group(1))
            break
    
    return {
        'val_accuracy': accuracy,
        'test_accuracy': test_accuracy,
        'last_epoch': last_epoch
    }

def parse_hyperparameter_value(filename, param_type):
    """Extract hyperparameter value from filename."""
    if param_type == 'num_examples':
        match = re.search(r'k_(\d+)', filename)
        return int(match.group(1)) if match else None
    elif param_type == 'learning_rate':
        # Handle scientific notation: lr_1e_5 -> 1e-5
        match = re.search(r'lr_([\d_]+(?:e_[\d_]+)?)', filename)
        if match:
            val_str = match.group(1).replace('_', '-') 
            if '-' in val_str and 'e' not in val_str:
                # Pattern like "1-5" should become "1e-5"
                parts = val_str.split('-', 1)
                if len(parts) == 2:
                    val_str = f"{parts[0]}e-{parts[1]}"
            return float(val_str)
        return None
    elif param_type == 'hidden_size':
        match = re.search(r'hidden_(\d+)', filename)
        return int(match.group(1)) if match else None
    elif param_type == 'batch_size':
        match = re.search(r'batch_(\d+)', filename)
        return int(match.group(1)) if match else None
    elif param_type == 'entropy':
        match = re.search(r'entropy_([\d_]+)', filename)
        if match:
            val_str = match.group(1).replace('_', '.')
            return float(val_str)
        return None
    elif param_type == 'gamma':
        match = re.search(r'gamma_([\d_]+)', filename)
        if match:
            val_str = match.group(1).replace('_', '.')
            return float(val_str)
        return None
    return None

def analyze_sensitivity(model_size='0.5B'):
    """Analyze all sensitivity experiments for a given model size."""
    results_dir = Path('results/sensitivity')
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return
    
    # Find all log files for this model size
    log_files = list(results_dir.glob(f'{model_size}_*.log'))
    
    if not log_files:
        print(f"No sensitivity results found for {model_size}")
        return
    
    # Organize results by hyperparameter type
    results = defaultdict(list)
    
    for log_file in log_files:
        filename = log_file.stem
        metrics = extract_epoch_results(str(log_file))
        
        if metrics is None or metrics['val_accuracy'] is None:
            continue
        
        # Determine hyperparameter type and value
        if 'k_' in filename:
            param_type = 'num_examples'
            value = parse_hyperparameter_value(filename, param_type)
        elif 'lr_' in filename:
            param_type = 'learning_rate'
            value = parse_hyperparameter_value(filename, param_type)
        elif 'hidden_' in filename:
            param_type = 'hidden_size'
            value = parse_hyperparameter_value(filename, param_type)
        elif 'batch_' in filename:
            param_type = 'batch_size'
            value = parse_hyperparameter_value(filename, param_type)
        elif 'entropy_' in filename:
            param_type = 'entropy'
            value = parse_hyperparameter_value(filename, param_type)
        elif 'gamma_' in filename:
            param_type = 'gamma'
            value = parse_hyperparameter_value(filename, param_type)
        else:
            continue
        
        if value is not None:
            results[param_type].append({
                'value': value,
                'val_accuracy': metrics['val_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'filename': filename
            })
    
    # Sort and print results
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS RESULTS - {model_size}")
    print(f"{'='*80}\n")
    
    for param_type in ['num_examples', 'learning_rate', 'hidden_size', 'batch_size', 'entropy', 'gamma']:
        if param_type not in results:
            continue
        
        data = results[param_type]
        data.sort(key=lambda x: x['value'])
        
        print(f"\n{param_type.upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"{'Value':<20} {'Val Accuracy':<15} {'Test Accuracy':<15}")
        print("-" * 80)
        
        best_val = max(data, key=lambda x: x['val_accuracy'])
        worst_val = min(data, key=lambda x: x['val_accuracy'])
        
        for item in data:
            marker = " <-- BEST" if item == best_val else " <-- WORST" if item == worst_val else ""
            test_acc_str = f"{item['test_accuracy']:.2f}%" if item['test_accuracy'] else "N/A"
            print(f"{str(item['value']):<20} {item['val_accuracy']*100:>6.2f}%{'':<6} {test_acc_str:<15}{marker}")
        
        # Calculate sensitivity (range)
        acc_range = best_val['val_accuracy'] - worst_val['val_accuracy']
        acc_range_pct = acc_range * 100
        print(f"\nRange: {acc_range_pct:.2f}% ({worst_val['val_accuracy']*100:.2f}% - {best_val['val_accuracy']*100:.2f}%)")
        
        # Calculate relative sensitivity
        mean_acc = sum(x['val_accuracy'] for x in data) / len(data)
        relative_sensitivity = (acc_range / mean_acc) * 100 if mean_acc > 0 else 0
        print(f"Relative Sensitivity: {relative_sensitivity:.1f}% (range/mean)")
    
    # Save results to JSON
    output_file = results_dir / f'sensitivity_summary_{model_size}.json'
    output_data = {}
    for param_type, data in results.items():
        output_data[param_type] = [
            {
                'value': item['value'],
                'val_accuracy': item['val_accuracy'],
                'test_accuracy': item['test_accuracy']
            }
            for item in sorted(data, key=lambda x: x['value'])
        ]
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    model_size = sys.argv[1] if len(sys.argv) > 1 else '0.5B'
    analyze_sensitivity(model_size)

