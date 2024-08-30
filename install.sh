#!/bin/bash
pip install -e ".[torch,metrics]"
pip install wandb bitsandbytes openai
pip install --upgrade huggingface_hub
huggingface-cli login --token "$PIPERW_HF_TOKEN"
pip install -r tmp_requirements.txt
