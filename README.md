# Efficient Split Learning with Overlapping Areas: Handling Distribution Shift in Multi-Cell Networks

This repository contains the implementation of **SplitOMC**, a framework for efficient split learning in multi-cell wireless networks that handles distribution shifts. The code is based on the paper:

"Efficient Split Learning with Overlapping Areas: Handling Distribution Shift in Multi-Cell Networks"\
Ati Rizwan, Dong-Jun Han, Md Ferdous Pervej, Christopher G. Brinton, Andreas F. Molisch, and Minseok Choi

## Description

SplitOMC leverages split learning to balance personalization and generalization in federated learning scenarios, particularly in multi-cell environments with overlapping regions. It addresses multi-level distribution shifts (main tasks, out-of-preference (OOP), and out-of-region (OOR) tasks) while reducing training and inference latency.

The code supports datasets like MNIST, CIFAR-10, and CIFAR-100, with options for non-IID data distributions using Dirichlet or random class assignments.

Two variants are implemented:

- `main_SplitOMC.py`: SplitOMC without network-wide aggregation.
- `main_SplitOMC+.py`: SplitOMC+ version with inter-edge server aggregation.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Torchvision
- NumPy
- Pandas
- Matplotlib
- tqdm
- scikit-learn

Install dependencies:

```
pip install torch torchvision numpy pandas matplotlib tqdm scikit-learn
```

## Usage

1. Configure the experiment in `config.json`:

   ```json
   {
       "dataset": "cifar10", 
       "clients": 50, 
       "existing_steup": true,
       "edge_servers": 5, 
       "overlap_percentage": 50, 
       "big_lambda": 0.5, 
       "lamda": 0.2, 
       "alpha": 0.4,
       "NonIID": 1
   }
   ```

   - `dataset`: "mnist", "cifar10", or "cifar100".
   - `clients`: Number of clients.
   - `edge_servers`: Number of edge servers.
   - `overlap_percentage`: Percentage of overlapping clients.
   - `big_lambda`: Hyperparameter for server aggregation.
   - `lamda`: Hyperparameter for client personalization.
   - `alpha`: Dirichlet concentration for non-IID (ND2).
   - `NonIID`: 1 for ND1 (random classes), 2 for ND2 (Dirichlet).

2. Run SplitOMC:

   ```
   python main_SplitOMC.py
   ```

3. Run SplitOMC+:

   ```
   python main_SplitOMC+.py
   ```

Results are saved as CSV files, e.g., `Alpha: 0.4 Lambda: 0.2 Overlapping: 50_Dataset: cifar10 Results.csv`.

## Code Structure

- `config.json`: Experiment configuration.
- `main_SplitOMC.py`: Main script for SplitOMC.
- `main_SplitOMC+.py`: Main script for SplitOMC+.
- `CreateZones.py`: Generates client-edge server mappings.
- `Client.py`: Client-side training and inference.
- `DataManager.py`: Handles dataset loading and non-IID partitioning.
- `Evaluator.py`: Model evaluation with entropy-based early exits.
- `Models.py`: Neural network models (AlexNet variants).
- `EdgeServer.py`: Edge server aggregation logic.
- `Network.py`: Network-wide aggregation for SplitOMC+.

## Citation

If you use this code, please cite: Will be updated after publication

