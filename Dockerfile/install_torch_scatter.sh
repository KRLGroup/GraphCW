#!/bin/bash

# Install torch-scatter with CUDA support
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-2.2.0+cu123.html

# Verify the installation
python -c "import torch; import torch_scatter; print(torch.cuda.is_available())"
