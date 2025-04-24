#!/bin/bash

ENV_NAME="venv"  # venv
KERNEL_NAME="venv-07" #
DISPLAY_NAME="Python (venv07)"

# Create venv if not exists
if [ ! -d "$ENV_NAME" ]; then
  echo "Creating virtual environment..."
  uv venv "$ENV_NAME"
fi

# Activate venv
source "$ENV_NAME/bin/activate"

# Install ipykernel
echo "Installing ipykernel..."
uv pip install ipykernel

# Register kernel
echo "Registering kernel with Jupyter..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$DISPLAY_NAME"

echo "Done! You can now select '$DISPLAY_NAME' in Jupyter Notebook."