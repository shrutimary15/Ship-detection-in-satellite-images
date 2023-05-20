# Ship Detection in Satellite Images using Deep Learning - 
[Master's Thesis]

This repository contains the implementation and documentation of my thesis project on ship detection in satellite images using deep learning techniques. The project includes a comparative analysis of three state-of-the-art models: YOLOv3, SSD, and Faster R-CNN. Among these models, YOLOv3 has demonstrated superior performance in ship detection tasks. The repository also includes the thesis document and relevant images.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Initial Exploration](#initial-exploration)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

### Introduction
Ship detection in satellite images plays a crucial role in various applications, such as maritime surveillance, environmental monitoring, and ship traffic analysis. The goal of this thesis project is to explore deep learning-based methods for ship detection and evaluate their performance. The focus is on comparing three popular models: YOLOv3, SSD, and Faster R-CNN, to identify the most effective solution for ship detection.

### Models
In this project, we have implemented and compared the following deep learning models:

- [YOLOv3](https://pjreddie.com/darknet/yolo/): YOLO (You Only Look Once) is an object detection algorithm known for its speed and accuracy. YOLOv3 uses a single neural network to simultaneously predict object bounding boxes and their class probabilities. 

- [SSD](https://www.kaggle.com/code/abedi756/ssd-single-shot-detection) (Single Shot MultiBox Detector): SSD is another popular object detection algorithm that achieves real-time object detection using a single neural network. It employs a series of convolutional layers with different scales to detect objects at multiple resolutions.

- [Faster R-CNN](https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/) (Region-based Convolutional Neural Network): Faster R-CNN is a two-stage object detection model. It uses a region proposal network (RPN) to generate potential object regions and then classifies and refines the proposals.

### Initial Exploration
Before diving into deep learning models, we initially explored using binary classification with the selective search algorithm for ship detection. Selective search is a region proposal algorithm that generates potential object regions in an image. We trained a binary classifier to classify these regions as ships or non-ships. However, this approach did not yield satisfactory results in terms of accuracy and efficiency, which led us to explore deep learning-based models for ship detection.

### Dataset
To train and evaluate the ship detection models, we used a open source dataset consisting of satellite images with ship annotations from the following [Github](https://github.com/amanbasu/ship-detection/tree/master/dataset). The dataset was carefully curated and annotated to ensure accurate and reliable training and testing. Details about the dataset, including its size, composition, and annotation format, are provided in the github account. 

For the initial explorations the dataset used for binary classifier can be found in this [link](https://www.kaggle.com/datasets/apollo2506/satellite-imagery-of-ships) and for applying selective search algorithm can be foungd in this [link](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery).

### Results
Based on the comparative analysis conducted in this thesis project, YOLOv3 has outperformed SSD and Faster R-CNN in ship detection tasks. The results, including MAP(Mean Average Precision) and speed of detection are detailed in the thesis document. Additionally, visualizations and comparisons of the model outputs are provided using sample images from the dataset in the thesis.

### Usage
To train and evaluate the ship detection models, follow the instructions provided in the thesis document. Typically, the following steps are involved:
1. Preprocess the dataset, including data augmentation and splitting into training and testing sets.
2. Configure the model-specific settings, such as hyperparameters and network architectures.
3. Train the models using the prepared dataset and chosen parameters.
4. Evaluate the trained models using appropriate evaluation metrics and testing datasets.
The specific commands and scripts required for each step are provided in the thesis document.

### Contributing
Contributions to this repository are welcome. If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request. Your contributions will be greatly appreciated.

### License
This project is licensed under the MIT License. Please review the license file for more information.

Please refer to the thesis document for more detailed instructions, explanations, and analysis related to ship detection in satellite images using deep learning models.







