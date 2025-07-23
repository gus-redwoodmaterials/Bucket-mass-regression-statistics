# Calciner Data Analysis

This project analyzes calciner data from Athena to understand the relationship between conveyor feed rates and bucket mass estimation residuals.

## Features

- **Data Caching**: Downloads data from Athena and caches locally as CSV for faster subsequent runs
- **Flexible Plotting**: Creates separate, clear visualizations for different aspects of the system
- **Residual Analysis**: Analyzes bucket mass estimation errors and their patterns
- **Correlation Analysis**: Studies the relationship between actual conveyor speed and estimation residuals

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install required packages:
```bash
pip install pandas matplotlib numpy pytz python-dateutil scipy
pip install rw-data-science-utils  # Custom Redwood package
```

## Usage

### Basic Usage
```bash
python 20250616_to_first_warmup.py
```

### Command Line Flags

- `--refresh` or `-r`: Fetch fresh data from Athena (ignores cache)
- `--plot` or `-p`: Display all plots
- `--analysis` or `-a`: Perform advanced residual correlation analysis

### Examples

```bash
# Display plots using cached data
python 20250616_to_first_warmup.py -p

# Fetch fresh data and run full analysis
python 20250616_to_first_warmup.py -r -a

# Run all features
python 20250616_to_first_warmup.py -r -p -a
```

## Analysis Features

### Basic Plots
1. **Bucket Mass Data**: Shows actual vs. predicted bucket mass
2. **Kiln Operations**: Weight averages, setpoints, and RPM
3. **Conveyor Systems**: Actual vs. setpoint conveyor speeds
4. **System Status**: ACME and robot status indicators
5. **Residual Analysis**: Error patterns and distributions

### Advanced Analysis (`-a` flag)
- **Peak Detection**: Identifies peaks in actual conveyor speed signal
- **Correlation Analysis**: Statistical correlation between feed rate and residuals
- **High Residual Detection**: Identifies periods of poor estimation performance
- **Feed Rate Bias Analysis**: Determines if estimator performs worse at high or low feed rates

## Data Sources

The analysis uses the following Athena tags:
- `bucket_mass`: Actual bucket mass measurements
- `bucket_mass_regress`: Regression model predictions
- `bucket_mass_resid`: Residuals (errors)
- `infeed_hz_actual`: Actual conveyor speed
- `infeed_hz_sp`: Conveyor speed setpoint
- Additional process variables (kiln weight, RPM, cycle times, etc.)

## Output

- **Console**: Statistical summaries and analysis results
- **Plots**: Interactive matplotlib visualizations
- **Cache**: CSV files stored in `data/` folder for faster reloading

## Time Range

Currently configured for July 20-21, 2025 (adjust `START` and `END` variables in the script as needed).

## Dependencies

- pandas: Data manipulation
- matplotlib: Plotting
- numpy: Numerical computations
- scipy: Peak detection and statistical analysis
- pytz: Timezone handling
- rw-data-science-utils: Athena data access (Redwood internal package)
