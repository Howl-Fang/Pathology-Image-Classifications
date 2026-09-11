#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
from datetime import datetime

def check_task_progress(task_dir, task_name, csv_name):
    """Check if a task has completed"""
    metrics_file = os.path.join(task_dir, 'metrics.json')
    csv_file = os.path.join(task_dir, f'{csv_name}.csv')
    
    result = {
        'task': task_name,
        'completed': False,
        'metrics': None,
        'csv_ready': False,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check for metrics
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file) as f:
                result['metrics'] = json.load(f)
                result['completed'] = True
        except:
            pass
    
    # Check for CSV
    if os.path.exists(csv_file):
        result['csv_ready'] = True
    
    return result

def main():
    base_path = "/storage/jmabq/Users/howl/First try"
    tasks = [
        ('task1_output', 'Task 1: CRC-MSI', 'CRC-MSI_prediction'),
        ('task2_output', 'Task 2: LUAD vs LUSC', 'NSCLC_prediction'),
        ('task3_output', 'Task 3: IM Detection', 'IM')
    ]
    
    print("\n" + "="*60)
    print("PATHOLOGY AI PROJECT - MONITORING")
    print("="*60)
    
    all_completed = True
    for task_dir, task_name, csv_name in tasks:
        path = os.path.join(base_path, task_dir)
        result = check_task_progress(path, task_name, csv_name)
        
        status = "✓ COMPLETED" if result['completed'] else "⏳ IN PROGRESS"
        print(f"\n{task_name}")
        print(f"  Status: {status}")
        
        if result['metrics']:
            print(f"  Metrics:")
            for metric, value in result['metrics'].items():
                print(f"    {metric}: {value:.4f}")
        
        if not result['completed']:
            all_completed = False
            log_file = os.path.join(path, 'training.log')
            if os.path.exists(log_file):
                with open(log_file) as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        print(f"  Last log: {lines[-1].strip()[:70]}...")
                    print(f"  Total log lines: {len(lines)}")
    
    print("\n" + "="*60)
    if all_completed:
        print("✓ ALL TASKS COMPLETED!")
    else:
        print("⏳ Tasks are still running. Check again later...")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
