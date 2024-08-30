#!/bin/bash

# Setup Personal OPENAI key
export AZURE_OPENAI_KEY_35=

export AZURE_OPENAI_ENDPOINT_35=

export AZURE_OPENAI_API_VERSION_35="2023-05-15"

export AZURE_OPENAI_KEY_4=

export AZURE_OPENAI_ENDPOINT_4=

export AZURE_OPENAI_API_VERSION_4="2023-05-15"

if ! [ -d 'emotion_metric_venv' ]; then
    python3 -m venv emotion_metric_venv
fi
source emotion_metric_venv/bin/activate


# Add current directory to PYTHONENV
python3 -c 'import sys; import os; sys.path.append(os.getcwd())'

# Install dependencies
python3 -m pip install -r requirements.txt
