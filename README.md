# Polynomial Regression from Scratch

A simple implementation of **Polynomial Regression from scratch** using Python and NumPy.

This project demonstrates how polynomial regression can be implemented without using machine learning libraries such as Scikit-learn. The model is trained using **Gradient Descent** and evaluated using **MAE** and **R²** metrics.

## Project Overview

In this project, Polynomial Regression is used to predict **MPG (Miles Per Gallon)** based on **HorsePower**.

The implementation includes:

* Polynomial feature generation
* Linear regression prediction
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² score
* Gradient calculation
* Gradient Descent optimization
* Training and testing
* Error history visualization

## Dataset

The project uses preprocessed training and testing datasets:

* `auto-train-preprocessed.csv`
* `auto-test-preprocessed.csv`

The main features used in the dataset are:

* **HorsePower** — Input feature
* **MPG** — Target variable

The training set contains **352 samples**, while the test set contains **40 samples**.

## Polynomial Regression

Polynomial features are generated using a custom function:

```python
def phi(x, d):
    phi_x = np.array([x**i for i in range(d+1)])
    phi_x = phi_x[..., 0].T
    return phi_x
```

For this project, the polynomial degree is set to:

```python
d = 3
```

Therefore, the model uses polynomial terms up to degree 3.

## Gradient Descent

The model parameters are optimized using Gradient Descent.

The training configuration is:

```python
eta = 0.1
epochs = 2000
```

During training, the model:

1. Calculates predictions.
2. Calculates the MAE loss.
3. Computes the gradients.
4. Updates the weights.
5. Stores the error history.

The error history is then plotted to observe the training process.

## Evaluation

After training, the model is evaluated on the test dataset.

The final results obtained in the notebook are:

| Metric   | Result |
| -------- | -----: |
| MAE      | 0.0833 |
| R² Score | 0.7909 |

These metrics are calculated using custom NumPy implementations rather than Scikit-learn.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

## Project Structure

```text
Polynomial-Regression/
│
├── Polynomial_Regression.ipynb
├── auto-train-preprocessed.csv
├── auto-test-preprocessed.csv
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib jupyter
```

### 3. Open the notebook

```bash
jupyter notebook Polynomial_Regression.ipynb
```

### 4. Run the cells

Make sure the following files are in the same directory as the notebook:

```text
auto-train-preprocessed.csv
auto-test-preprocessed.csv
```


