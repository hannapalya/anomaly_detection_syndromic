# Google Colab Setup Instructions for LSTM.ipynb

## Required Files to Upload

Upload these files to Google Colab to run `LSTM.ipynb`:

### 1. Python Module Files (Required)
- `r_comparator_metrics.py` - Contains R-comparator metric functions
- `load_split_indices.py` - Helper module for loading predefined splits

### 2. Configuration File (Required)
- `train_val_test_split.json` - Contains predefined train/validation/test split indices

### 3. Data Files (Required)
Upload the entire `signal_datasets_large/` directory (or whichever directory contains your data), specifically:
- `simulated_totals_sig1.csv` through `simulated_totals_sig9.csv`
- `simulated_outbreaks_sig1.csv` through `simulated_outbreaks_sig9.csv`

**Note:** The notebook processes signals 1-9 (as defined by `SIGNALS = list(range(1, 10))`), so you need data files for those signals.

### 4. The Notebook (Required)
- `LSTM.ipynb` - The main notebook to run

## Upload Process

### Option 1: Upload via Colab UI
1. Open Google Colab and create a new notebook or open `LSTM.ipynb`
2. Click the folder icon in the left sidebar to open the file browser
3. Upload each file:
   - `r_comparator_metrics.py`
   - `load_split_indices.py`
   - `train_val_test_split.json`
   - The entire `signal_datasets_large/` folder (or create the folder and upload CSV files into it)

### Option 2: Use Google Drive
1. Upload all files to Google Drive
2. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update `DATA_DIR` in the notebook to point to your Drive folder:
   ```python
   DATA_DIR = "/content/drive/MyDrive/path/to/signal_datasets_large"
   ```

### Option 3: Use GitHub (if repository is public)
1. Clone the repository directly in Colab:
   ```python
   !git clone https://github.com/yourusername/anomaly_detection_syndromic.git
   ```
2. Update `DATA_DIR` if needed:
   ```python
   DATA_DIR = "anomaly_detection_syndromic/signal_datasets_large"
   ```

## Quick Setup Cell for Colab

Add this cell at the beginning of your notebook to install dependencies:

```python
# Install required packages (if not already installed)
!pip install tensorflow pandas numpy scikit-learn

# Verify files are present
import os
print("Checking for required files...")
print(f"r_comparator_metrics.py: {os.path.exists('r_comparator_metrics.py')}")
print(f"load_split_indices.py: {os.path.exists('load_split_indices.py')}")
print(f"train_val_test_split.json: {os.path.exists('train_val_test_split.json')}")
print(f"Data directory: {os.path.exists('signal_datasets_large') or os.path.exists('signal_datasets')}")
```

## Important Notes

1. **Data Directory**: The notebook uses `DATA_DIR = ""` which means it looks for CSV files in the current directory. Make sure your CSV files are in the correct location relative to where the notebook runs.

2. **GPU**: The notebook will automatically use GPU if available in Colab. You can enable GPU in Colab by going to: Runtime → Change runtime type → Hardware accelerator → GPU

3. **File Structure**: After uploading, your Colab file structure should look like:
   ```
   /content/
   ├── LSTM.ipynb
   ├── r_comparator_metrics.py
   ├── load_split_indices.py
   ├── train_val_test_split.json
   └── signal_datasets_large/  (or signal_datasets/)
       ├── simulated_totals_sig1.csv
       ├── simulated_outbreaks_sig1.csv
       ├── simulated_totals_sig2.csv
       ├── simulated_outbreaks_sig2.csv
       └── ... (for signals 1-9)
   ```

## Verifying the Setup

Run this cell to verify everything is set up correctly:

```python
import os
import json

# Check Python modules
try:
    from r_comparator_metrics import compute_sensitivity_R, IDX_RANGE
    from load_split_indices import get_signal_split
    print("✓ Python modules loaded successfully")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")

# Check split file
if os.path.exists('train_val_test_split.json'):
    with open('train_val_test_split.json', 'r') as f:
        data = json.load(f)
        print(f"✓ Split file loaded: {len(data.get('by_signal', {}))} signals")
else:
    print("✗ train_val_test_split.json not found")

# Check data files
DATA_DIR = "signal_datasets_large"  # or "signal_datasets" if you use that
if not os.path.exists(DATA_DIR):
    DATA_DIR = "signal_datasets"  # try alternative

if os.path.exists(DATA_DIR):
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    totals = [f for f in csvs if 'totals' in f and any(f'sig{i}' in f for i in range(1, 10))]
    outbreaks = [f for f in csvs if 'outbreaks' in f and any(f'sig{i}' in f for i in range(1, 10))]
    print(f"✓ Data directory found: {DATA_DIR}")
    print(f"  Totals files (sig1-9): {len(totals)}")
    print(f"  Outbreaks files (sig1-9): {len(outbreaks)}")
else:
    print(f"✗ Data directory not found. Please upload signal_datasets_large/ or signal_datasets/")
```


