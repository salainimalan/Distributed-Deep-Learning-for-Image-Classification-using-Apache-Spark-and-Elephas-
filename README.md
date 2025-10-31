# Distributed Deep Learning for Image Classification using Apache Spark and Elephas 
# Distributed Deep Learning for Image Classification using Apache Spark and Elephas

This project demonstrates how to build and train a distributed deep learning model for image classification using Apache Spark and Elephas on the Caltech-101 dataset.  
By integrating Spark’s distributed data processing with Elephas’s scalable Keras training, the project achieves efficient and parallelized image model training across cluster nodes.

---

## Project Overview

The goal of this project is to perform image classification (Cats vs Dogs) using a Spark + Elephas architecture.  
The workflow covers the entire deep learning lifecycle — from image preprocessing to distributed model training and evaluation.

**Key Objective:**  
Leverage Elephas to enable distributed training of Keras models directly on Spark RDDs, reducing computation time and scaling model performance.

---

## Workflow

### 1. Spark Initialization
- Created a SparkSession with 4 GB driver memory and local parallel processing.  
- Managed distributed computation using SparkContext for scalable data handling.

### 2. Image Preprocessing
- Loaded raw images from the Caltech-101 dataset.  
- Resized to 128×128, converted to grayscale, and applied Gaussian denoising.  
- Applied data augmentation (flipping, rotation) to improve dataset diversity.

### 3. Data Preparation in Spark
- Flattened image matrices into feature vectors.  
- Created a Spark DataFrame with `features` and `label` columns.  
- Converted Spark DataFrame into an RDD using Elephas utilities.

### 4. Distributed Model Training
- Defined a Keras Sequential model with dense and dropout layers.  
- Wrapped it with Elephas SparkModel for distributed training.  
- Used asynchronous gradient updates for faster convergence.  
- Achieved a test accuracy of 71.34% after 5 epochs.

---

## Role and Benefits of Elephas

Elephas bridges Keras and Apache Spark, enabling deep learning at scale.

**Why Elephas?**
- Scalability: Distributes training across multiple Spark executors.  
- Asynchronous Updates: Reduces synchronization delays between workers.  
- Fault Tolerance: Inherits Spark’s resilience to node failures.  
- Easy Integration: Works directly with Keras models and RDDs.  
- Speed: Trains large models on massive datasets without centralized bottlenecks.

---

## Results

| Metric | Value |
|:-------|:------:|
| Dataset | Caltech-101 (Cats vs Dogs subset) |
| Input Size | 128×128 (Grayscale) |
| Accuracy | 71.34% |
| Mode | Asynchronous Distributed Training |
| Frameworks Used | Apache Spark, Keras, Elephas |

---

## Technologies Used

- Apache Spark – Distributed data processing  
- Elephas – Distributed deep learning on Spark  
- Keras / TensorFlow – Model definition and training  
- Python (PIL, NumPy, SciPy) – Image preprocessing  
- Matplotlib – Visualization  

---

## Example Model Architecture

```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(64*64*3,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
