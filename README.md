# Customer Churn Prediction with TensorFlow

This repository contains a machine learning project for predicting customer churn using TensorFlow. The project demonstrates the complete workflow, from data preprocessing and model training to evaluation and deployment. It includes the dataset, Jupyter Notebook for implementation, and the trained `.h5` model file.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Files in this Repository](#files-in-this-repository)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

Customer churn refers to the loss of customers over time. Predicting churn is critical for businesses aiming to retain customers and reduce costs. This project uses TensorFlow to create a predictive model that identifies customers likely to churn based on historical data.

### Key Features:
- Data preprocessing: Handles missing values, encoding categorical features, and feature scaling.
- Model training: Implements a deep learning model using TensorFlow/Keras.
- Model evaluation: Includes metrics like accuracy, precision, recall, and F1-score.
- Saved model: The trained model is saved as an `.h5` file for easy deployment.

---

## Files in this Repository

1. **`customer_churn_prediction.ipynb`**  
   The Jupyter Notebook with the code for data preprocessing, model training, evaluation, and visualization.
   
2. **`dataset.csv`**  
   The dataset used for training and evaluating the model. It includes customer details, usage statistics, and churn labels.

3. **`trained_model.h5`**  
   The trained TensorFlow model saved in the `.h5` format for deployment and reuse.

---

## Dataset

The dataset is a CSV file (`dataset.csv`) containing customer information such as:  
- Demographic details (e.g., age, gender)
- Account and service details (e.g., subscription type)
- Churn status (target variable)

### Columns:
- `Age`, `Gender` etc.
- `Churn`: Binary target variable (`1` for churn, `0` for retention).

---

## Model Architecture

The TensorFlow model is a feed-forward neural network with:
- Input layer: Handles feature input after preprocessing.
- Hidden layers: Fully connected layers with ReLU activation.
- Output layer: A single neuron with a sigmoid activation for binary classification.

---

## Getting Started

### Prerequisites
Ensure you have Python 3.7+ and the following libraries installed:
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/TawanaState/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook customer_churn_prediction.ipynb
   ```

3. Explore the dataset, preprocess data, train the model, and evaluate its performance.

4. Use the saved `.h5` model for predictions:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('trained_model.h5')
   predictions = model.predict(new_data)
   ```

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.  

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
