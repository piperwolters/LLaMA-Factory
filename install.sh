#!/bin/bash
pip install -e ".[torch,metrics]"
pip install wandb bitsandbytes openai
pip install --upgrade huggingface_hub
huggingface-cli login --token "$PIPERW_HF_TOKEN"
wandb login "$PIPERW_WANDB_TOKEN"
pip install -r requirements.txt
python setup.py install
export WANDB_PROJECT='osagent'
