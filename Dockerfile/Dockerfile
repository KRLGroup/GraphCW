# Docker file for PyTorch

FROM nvcr.io/nvidia/pytorch:23.12-py3

USER root

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    libxrender1 \
    python3-rdkit

# Install required Python packages
RUN pip install hydra-core==1.2.0 \
    numpy==1.23.0 \
    tqdm==4.64.0 \
    omegaconf==2.2.2 \
    captum==0.5.0 \
    networkx==2.8.2 \
    matplotlib==3.5.2 \
    seaborn==0.13.2 \
    scikit-image==0.24.0

# Install PyTorch Geometric dependencies
RUN pip install torch-geometric==2.1.0 \
    torch_sparse==0.6.13 \
    dive-into-graphs==0.1.2 \
    pylablib==1.4.0

# Ensure CUDA environment variables are set
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Install torch-scatter interactively after container starts
COPY install_torch_scatter.sh /opt/install_torch_scatter.sh
RUN chmod +x /opt/install_torch_scatter.sh

# Use entrypoint to ensure correct environment setup
ENTRYPOINT ["/bin/bash", "-c", "/opt/install_torch_scatter.sh && bash"]
