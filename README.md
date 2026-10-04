# Solar Panel Fault Detection Using YOLO11 and Federated Learning

## Overview

This project presents an AI-based solar panel fault detection system using **YOLO11**, enhanced with **BiFPN-CBAM** and **Adaptive Spatial Feature Fusion (ASFF)** for improved feature extraction and multi-scale defect detection. The model achieved **95% mAP** in solar panel defect detection.

The project also implements **Federated Learning (FL)** using two clients and one central server with **Federated Batch Normalization (FedBN)**. This approach enables collaborative model training while preserving client-specific Batch Normalization statistics and aggregating model weights, helping maintain local feature distributions across different clients.

## Key Features

- **YOLO11-Based Detection:** Detects solar panel defects using an object detection architecture.
- **BiFPN:** Enhances multi-scale feature fusion through bidirectional feature propagation.
- **CBAM Attention:** Uses channel and spatial attention to emphasize relevant defect features.
- **ASFF:** Adaptively fuses features from different scales to improve detection performance.
- **95% mAP:** Achieved a mean Average Precision of 95% in model evaluation.
- **Federated Learning:** Implements a distributed training setup with two clients and one central server.
- **FedBN:** Preserves local Batch Normalization statistics while aggregating model weights across clients.
- **Privacy-Aware Training:** Supports collaborative learning without requiring central collection of raw client training images.

## System Architecture

### 1. YOLO11 with Enhanced Feature Fusion

The object detection pipeline incorporates:

- **YOLO11:** Performs solar panel defect localization and classification.
- **BiFPN:** Combines features across multiple resolutions.
- **CBAM:** Applies channel and spatial attention to highlight informative features.
- **ASFF:** Adaptively combines multi-scale feature maps for defect detection.

### 2. Federated Learning with FedBN

The federated training system consists of:

- **Client 1:** Trains the local model using its private dataset.
- **Client 2:** Trains the local model using its private dataset.
- **Central Server:** Coordinates training rounds and aggregates eligible model weights.
- **FedBN:** Keeps Batch Normalization parameters and running statistics local to each client while aggregating the remaining eligible model parameters.

**Federated training workflow:**

1. The server initializes the global model.
2. The server distributes the shared model parameters to both clients.
3. Each client trains the model on its local dataset.
4. Clients send eligible model weights or updates to the server.
5. The server aggregates the shared parameters.
6. The updated shared model is distributed to the clients for the next round.
7. Each client retains its local Batch Normalization statistics.

This approach supports decentralized training and can improve robustness when clients have different data distributions.

## Performance

| Metric | Result |
|---|---|
| Model | YOLO11 |
| Detection Performance | 95% mAP |
| Feature Fusion | BiFPN and ASFF |
| Attention Mechanism | CBAM |
| Federated Learning Clients | 2 |
| Central Servers | 1 |
| Federated Learning Strategy | FedBN |

*Note: The reported 95% mAP should be accompanied by the evaluation metric variant (such as mAP@0.5 or mAP@0.5:0.95), dataset, and evaluation split when those details are available.*

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Object Detection:** YOLO11
- **Feature Fusion:** BiFPN, ASFF
- **Attention Mechanism:** CBAM
- **Distributed Training:** Federated Learning
- **Federated Aggregation:** FedBN
- **Data Processing and Evaluation:** NumPy, OpenCV, and relevant model evaluation tools

*Adjust the technology list to match the libraries and tools actually used in the implementation.*

## Project Objectives

- Develop an accurate deep learning model for solar panel defect detection.
- Improve multi-scale feature representation using BiFPN and ASFF.
- Enhance defect-related feature learning through CBAM attention.
- Enable collaborative model training across decentralized clients.
- Preserve client-specific Batch Normalization statistics using FedBN.
- Reduce the need to centralize raw training data.

## Applications

- Solar farm inspection and maintenance
- Automated photovoltaic panel defect detection
- Large-scale solar installation monitoring
- Privacy-aware collaborative training across distributed inspection sites

## Future Improvements

- Evaluate the model on larger and more diverse solar panel datasets.
- Compare the enhanced model against the baseline YOLO11 architecture.
- Report precision, recall, F1-score, mAP@0.5, and mAP@0.5:0.95.
- Evaluate federated performance under non-IID client data distributions.
- Scale the federated system to more clients and communication rounds.
- Explore real-time deployment using edge devices.

## Author

**Nandana Thachilath**

GitHub: [@NandanaThachilath](https://github.com/NandanaThachilath)

---

If you find this project useful, consider starring the repository!
