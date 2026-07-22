#!/usr/bin/env bash
set -euo pipefail

dist=$(. /etc/os-release && echo $ID)
ver=$(. /etc/os-release && echo $VERSION_ID)

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
curl -s -L "https://nvidia.github.io/libnvidia-container/$dist$ver/libnvidia-container.list" | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update -qq
sudo apt-get install -y -qq nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "GPU support setup complete."
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi 2>&1 | head -5
