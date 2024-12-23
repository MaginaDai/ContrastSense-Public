# ContrastSense: Domain-invariant Contrastive Learning for In-the-wild Wearable Sensing

## Overview
ContrastSense is a domain-invariant contrastive learning framework designed for wearable sensing scenarios that face challenges including domain shifts and class label scarcity. The method improves robustness and generalizability across diverse domains by leveraging domain and time labels. 

---

## Installation
### Requirements
- Python >=3.7.13

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## Datasets
The framework has been evaluated on the following datasets:
- **Human Activity Recognition (HAR)**: HHAR, MotionSense, Shoaib, HASC-PAC2016.
- **Gesture Recognition (GR)**: MyoArmBand, NinaPro DB4, NinaPro DB5.

---

## Usage

### Dataset Preprocessing
To prepare the datasets for ContrastSense, follow these steps:

1. **Download the Datasets**  
   Place the raw datasets in the `./original_dataset` directory.

2. **Preprocess the Datasets**  
   Run the preprocessing scripts in `./data_preprocessing` for each dataset. Below are the commands to preprocess the supported datasets:

   ```bash
   # Preprocess HASC dataset
   python HASC.py

   # Preprocess HHAR dataset
   python HHAR.py

   # Preprocess MotionSense dataset
   python MotionSense.py

   # Preprocess Myo dataset
   python Myo.py

   # Preprocess NinaPro dataset
   python NinaPro.py

   # Preprocess Shoaib dataset
   python Shoaib.py

   # Preprocess UCI EMG dataset
   python UCI_emg.py
   ```

3. **Generate Dataset Splits**  

After preprocessing, generate dataset splits for different cross-domain scenarios using:
```bash
python data_split.py
```

To generate splits for different cross domain scenarios and datasets, you may setup the dataset, cross, split ratios in data_split.py.

### Pretraining
To pretrain a model using ContrastSense:
```bash
bash pretrain.sh
```

### Fine-tuning and Evaluation
To fine-tune the pretrained model:
```bash
bash finetune.sh
```

### Models for Mobile Phones
To convert the fine-tuned model to the mobile version:
```bash
python convert_from_torch_to_mobile_version.py
```

---

## Citing ContrastSense
If you use this code, please cite:
```

@article{ContrastSense2024,
    author = {Dai, Gaole and Xu, Huatao and Yoon, Hyungjun and Li, Mo and Tan, Rui and Lee, Sung-Ju},
    title = {ContrastSense: Domain-invariant Contrastive Learning for In-the-Wild Wearable Sensing},
    year = {2024},
    issue_date = {November 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {8},
    number = {4},
    url = {https://doi.org/10.1145/3699744},
    doi = {10.1145/3699744},
    journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
    month = nov,
    articleno = {162},
    numpages = {32},
}

```

---

## License
This project is licensed under the Creative Commons Attribution 4.0 International License.

---

## Acknowledgements
This project builds upon the following works:

- **[MoCo](https://github.com/facebookresearch/moco)**: Momentum Contrast for Unsupervised Visual Representation Learning.
- **[SimCLR](https://github.com/google-research/simclr)**: A Simple Framework for Contrastive Learning of Visual Representations.
- **[SupContrast](https://github.com/HobbitLong/SupContrast)**: Supervised Contrastive Learning.
- **[LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public)**: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications.

We thank the authors of these works for sharing their implementations, which formed the foundation for some aspects of ContrastSense.

---

## Contributing
Contributions are welcome!
