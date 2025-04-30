Authors: Ramneek Ahluwalia 

Description: This repository contains the code to implement a custom Mask R-CNN lti-task object detection and semantic segmentation model on the CamVid dataset

## Running Solution

This repository has the following structure 

.
├── data/                                       # Dataset directory
├── src/                                        # Source code
│   ├── pytorch_helper/                         # PyTorch utility scripts
│   │   ├── coco_eval.py                        # COCO evaluation metrics
│   │   ├── coco_utils.py                       # COCO utilities for evaluation
│   │   ├── engine.py                           # Training and evaluation utilities
│   │   ├── transforms.py                       # Transformations for data augmentation
│   │   └── utils.py                            # General utilities
│   ├── camvid_analysis.py                      # CamVid dataset analysis
│   ├── setup.py                                # Project setup script
│   ├── dataset_manager.py                      # File management utilities
│   ├── dataloader.py                           # Data loading and pre-processing utilities
│   ├── train.py                                # Training script
│   └── test.py                                 # Testing and evaluation script
├── results/                                    # Results directory
│   ├── evaluation_results/                     # Evaluation metrics and reports
│   ├── visualization_results/                  # Visualization outputs
│   └── histogram_based_augmentation.png        # Augmentation visualization
├── requirements.txt                            # Dependencies
└── README.md                                   # This file

Please ensure that your CamVId dataset is placed in the data folder and only run the setup.py file if you do not have the helper files already in the folder. 


## Visual Results

Some examples of the achieved predictions are shown below:


### Sample Model Predictions
<p align="center">
  <img src="/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/src/results/visualization_results/sample_0.png"width="80%" alt="Sample prediction"/>
</p>

<p align="center">
  <img src="/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/src/results/visualization_results/sample_1.png"width="80%" alt="Sample prediction"/>
</p>

<p align="center">
  <img src="/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/src/results/visualization_results/sample_2.png"width="80%" alt="Sample prediction"/>
</p>

<p align="center">
  <img src="/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/src/results/visualization_results/sample_7.png"width="80%" alt="Sample prediction"/>


## Acknowledgements
- Code snippets used from - COMP0248 - Week 4 accessible at: https://moodle.ucl.ac.uk/course/section.php?id=1056534
- CamVid dataset by University of Cambridge
- PyTorch Utility files 