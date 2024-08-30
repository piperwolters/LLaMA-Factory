#!/bin/bash
pip install -e ".[torch,metrics]"
pip install wandb bitsandbytes openai
pip install --upgrade huggingface_hub
echo -e "$PIPERW_HF_TOKEN" | huggingface-cli login --token

