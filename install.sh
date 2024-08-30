#!/bin/bash
pip install -e ".[torch,metrics]"
pip install wandb bitsandbytes openai
pip install --upgrade huggingface_hub
huggingface-cli login
