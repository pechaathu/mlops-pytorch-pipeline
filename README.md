# mlops-pytorch-pipeline

An end-to-end MLOps deployment pipeline taking a PyTorch image classifier
(ResNet-18 on CIFAR-10) through the full lifecycle: Git workflow, Docker
containerization, and Kubernetes orchestration.

## Project Structure

- `src/` — model, dataset, training, and serving code
- `configs/` — training configuration (YAML)
- `docker/` — Dockerfiles for training and serving images
- `k8s/` — Kubernetes manifests (Job, Deployment, Service, ConfigMap)
- `requirements/` — separate dependency lists for training and serving
- `tests/` — unit tests
- `.github/workflows/` — CI pipeline

## Status

🚧 Work in progress.

## Setup

Instructions to follow.
