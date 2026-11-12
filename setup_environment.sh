#!/bin/bash

# ConceptGraphs Environment Setup Script (国内镜像源加速版)
# 使用清华、阿里云等国内源，大幅提升下载速度

set -e

echo "========================================"
echo "ConceptGraphs Environment Setup"
echo "(China Mirror Accelerated)"
echo "========================================"

# 配置国内镜像源
echo ""
echo "[0/9] Configuring China mirror sources..."
# 添加清华conda镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

# 配置pip镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "[1/9] Creating conda environment 'conceptgraph'..."
conda create -n conceptgraph python=3.10 -y

echo ""
echo "[2/9] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate conceptgraph

echo ""
echo "[3/9] Installing basic libraries (using Tsinghua mirror)..."
pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy

echo ""
echo "[4/9] Installing YOLO..."
pip install ultralytics

echo ""
echo "[5/9] Installing Faiss (using Tsinghua mirror)..."
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y

echo ""
echo "[6/9] Installing PyTorch with CUDA 11.8 (using Tsinghua mirror)..."
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "[7/9] Installing PyTorch3D..."
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -y

echo ""
echo "[8/9] Installing gradslam and chamferdist (using mirror)..."

git clone https://github.com/krrish94/chamferdist.git
cd chamferdist && pip install . && cd ..

git clone https://github.com/gradslam/gradslam.git
cd gradslam && git checkout conceptfusion && pip install .

echo ""
echo "[9/9] Cleanup and verification..."
cd "$(dirname "$0")"
conda deactivate

