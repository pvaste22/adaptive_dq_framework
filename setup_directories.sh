#!/bin/bash
# Create all project directories

# Config and data directories
mkdir -p config data/{raw,processed/{training/{v0,v1,v2},windowed_data}}
mkdir -p data/adaptation_buffer/{window_buffer,consolidated,archive}
mkdir -p data/artifacts/{models,scalers,pca,distributions,baselines}
mkdir -p data/artifacts/{anomaly_detectors,historical_windows,matrices}
mkdir -p data/artifacts/{patterns,templates,thresholds}
mkdir -p data/logs/{training,deployment,adaptation}

# Source code directories
mkdir -p src/{common,data_processing,quality_dimensions,scoring}
mkdir -p src/{drift_detection,method_activator,training,deployment,xapp}

# Other directories
mkdir -p scripts analysis tests/{unit,integration}
mkdir -p xapp_descriptor docker

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
touch analysis/__init__.py tests/__init__.py

echo "✅ Directory structure created successfully!"
