import torch
import numpy as np
from itertools import product
import random
from load_data import load_data
from ncf import NCF
from train_ncf import train_ncf
from evaluate import evaluate
import json
from datetime import datetime
from collections import defaultdict
from statistics import mean

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Experimental configurations
path = 'ratebeer/'
print(f'Loading data from {path}')

# Define hyperparameter search space
embedding_configs = [
    {'use_social': False, 'use_time': False, 'giver': False},
    {'use_social': False, 'use_time': False, 'giver': True},
    {'use_social': True, 'use_time': False, 'giver': False},
    {'use_social': False, 'use_time': True, 'giver': False},
    {'use_social': True, 'use_time': True, 'giver': False},
]

embedding_dims = [8]
batch_sizes = [512]
epochs = [100]
learning_rates = [0.001]
hidden_layer_configs = [[2]]
NUM_RUNS = 1  # Number of runs for each configuration

# Initialize results storage
results = []
experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def config_to_key(config, params):
    return (
        frozenset(config.items()),
        params['embedding_dim'],
        params['batch_size'],
        params['epochs'],
        params['lr'],
        tuple(params['hidden_layers'])
    )

# Dictionary to store runs for each configuration
config_runs = defaultdict(list)

# Run experiments
for config in embedding_configs:
    print(f"\nTesting embedding configuration: {config}")
    
    try:
        # Load data with current configuration
        train_data, num_users, num_items, test_data, social_adj, C = load_data(
            path, **config
        )
        
        # Move data to device
        train_data = train_data.to(device)
        test_data = test_data.to(device)
        
        # Generate all combinations of hyperparameters
        param_combinations = list(product(
            embedding_dims,
            batch_sizes,
            epochs,
            learning_rates,
            range(len(hidden_layer_configs))
        ))
        
        for emb_dim, batch_size, epoch, lr, hidden_idx in param_combinations:
            current_params = {
                'embedding_dim': emb_dim,
                'batch_size': batch_size,
                'epochs': epoch,
                'lr': lr,
                'hidden_layers': hidden_layer_configs[hidden_idx]
            }
            
            print(f"\nStarting experiments for configuration:")
            print(f"Embedding config: {config}")
            print(f"Parameters: {current_params}")
            
            # Run multiple times for this configuration
            for run in range(NUM_RUNS):
                print(f"\nRun {run + 1}/{NUM_RUNS}")
                
                try:
                    # Initialize model
                    model = NCF(
                        num_users,
                        num_items,
                        embedding_dim=emb_dim,
                        hidden_layers=hidden_layer_configs[hidden_idx]
                    )
                    model = model.to(device)
                    
                    completed_epochs = train_ncf(
                        model,
                        train_data,
                        social_adj,
                        num_users,
                        num_items,
                        epochs=epoch,
                        batch_size=batch_size,
                        lr=lr,
                        use_time=config['use_time'],
                        use_giver=config['giver']
                    )

                    if completed_epochs > 0:
                        try:
                            auc = evaluate(
                                model,
                                test_data,
                                num_users,
                                num_items,
                                social_adj,
                                train_data, 
                                use_giver=config['giver'],
                                use_time=config['use_time']
                            )
                            print(f'Run {run + 1} AUC: {auc:.4f}')
                            config_key = config_to_key(config, current_params)
                            config_runs[config_key].append(auc)
                            
                        except Exception as e:
                            print(f"Evaluation failed: {str(e)}")
                            continue
                    else:
                        print("No successful training epochs completed, skipping evaluation")
                        
                except RuntimeError as e:
                    print(f"Runtime error in experiment: {str(e)}")
                    continue
            
            # After all runs for this configuration, calculate and store average
            config_key = config_to_key(config, current_params)
            valid_runs = config_runs[config_key]
            
            if valid_runs:
                avg_auc = mean(valid_runs)
                result = {
                    'embedding_config': config,
                    'embedding_dim': emb_dim,
                    'batch_size': batch_size,
                    'target_epochs': epoch,
                    'learning_rate': lr,
                    'hidden_layers': hidden_layer_configs[hidden_idx],
                    'avg_auc': float(avg_auc),
                    'num_valid_runs': len(valid_runs),
                    'valid_run_scores': valid_runs,
                    'std_dev': float(np.std(valid_runs)) if len(valid_runs) > 1 else 0.0
                }
                results.append(result)
                
                # Save results after each configuration
                with open(f'results_{experiment_time}.json', 'w') as f:
                    serializable_results = []
                    for r in results:
                        r_copy = r.copy()
                        if isinstance(r_copy['embedding_config'], dict):
                            r_copy['embedding_config'] = {
                                k: v if not isinstance(v, set) else list(v)
                                for k, v in r_copy['embedding_config'].items()
                            }
                        serializable_results.append(r_copy)
                    json.dump(serializable_results, f, indent=4)
                    
    except Exception as e:
        print(f"Error with embedding configuration {config}: {str(e)}")
        continue

# Print final summary
print("\nExperiment Summary:")
print(f"Total configurations tested: {len(results)}")
print("\nTop 5 configurations by average AUC:")
sorted_results = sorted(results, key=lambda x: x['avg_auc'], reverse=True)
for i, result in enumerate(sorted_results[:5]):
    print(f"\n{i+1}. Avg AUC: {result['avg_auc']:.4f} (std: {result['std_dev']:.4f})")
    print(f"Valid runs: {result['num_valid_runs']}/{NUM_RUNS}")
    print(f"Run scores: {[f'{score:.4f}' for score in result['valid_run_scores']]}")
    print(f"Configuration:")
    for key, value in result.items():
        if key not in ['avg_auc', 'num_valid_runs', 'valid_run_scores', 'std_dev']:
            print(f"  {key}: {value}")