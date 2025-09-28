<div align="center">
<img src="https://img.shields.io/badge/Project%20Status-Complete-FFD700?style=for-the-badge&logo=github" alt="Project Status Badge">
<img src="https://img.shields.io/badge/Model%20Type-Binary%20Classification-0077B6?style=for-the-badge&logo=pytorch" alt="Binary Classification Badge">
<img src="https://img.shields.io/badge/Algorithm-XGBoost%20%7C%20Scikit--learn-FF4500?style=for-the-badge&logo=scikitlearn" alt="XGBoost/Scikit-learn Badge">
</div>

<h1 align="center">
    <img src="https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/btc.png" alt="Bitcoin Logo" width="60" height="60" style="vertical-align: middle; margin-right: 10px;">
    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&center=true&vCenter=true&width=600&height=70&duration=3000&lines=₿%20Bitcoin%20Price%20Prediction;%20Machine%20Learning%20Analysis&color=FFD700" style="vertical-align: middle;" />
</h1>
<div align="center">
    </div>

---

## 🌊 Project Overview
This repository features a comprehensive **Machine Learning** project using **Python** and **Scikit-learn** to predict Bitcoin's price movement direction. The core focus is on transforming historical Bitcoin price data into powerful **classification models** capable of predicting whether Bitcoin's price will go **up or down** on any given trading day.

### 🎯 Key Deliverables

* **Classification Models:** Implementing Logistic Regression, SVC, and XGBClassifier for directional prediction.
* **Feature Engineering:** Creating moving averages, volatility measures, and technical indicators.
* **Performance Analysis:** Comprehensive model evaluation with accuracy, precision, recall, and confusion matrices.

---

## 💻 Repository Files & Tech Stack

### 📂 File Structure

| File Name | Data Type | Description |
| :--- | :--- | :--- |
| `bitcoin.csv` | **Historical Time-Series** | Daily Bitcoin price data (2014-2022) including Open, High, Low, Close, Adjusted Close, and Volume. |
| `Bitcoin-Price-Prediction-using-Machine-Learning-in-Python.ipynb` | **Analysis Notebook** | Core document containing data preprocessing, feature engineering, model training, and evaluation. |
| `README.md` | **Documentation** | This comprehensive guide to the project structure and methodology. |

### 🛠️ Key Technologies

| Category | Tools/Libraries | Purpose |
| :--- | :--- | :--- |
| **Data Handling** | `Pandas`, `NumPy` | Essential for data manipulation, cleaning, and numerical operations. |
| **Machine Learning** | `Scikit-learn`, `XGBoost` | Model selection, training, hyperparameter tuning, and evaluation. |
| **Visualization** | `Matplotlib`, `Seaborn` | Creating insightful charts for price trends, correlations, and model performance metrics. |

---

## 🧠 Step-by-Step Analysis Guide (The Notebook Flow)

This section details the objective and outcome of the main analytical steps within the Jupyter Notebook.

### Phase 1: Data Preparation and Feature Engineering

| Step | Aim | Achievement |
| :--- | :--- | :--- |
| **1. Data Loading & Cleaning** | Load Bitcoin historical data and handle missing values. Convert date columns and ensure price/volume data is numeric. | **Data Integrity.** Clean, chronologically ordered dataset ready for feature engineering. |
| **2. Target Variable Creation** | Calculate daily price movement direction (Binary: 1 for price increase, 0 for decrease). | **Classification Target.** Created the fundamental binary target for supervised learning. |
| **3. Technical Indicators** | Generate moving averages (MA), volatility measures (High-Low spread), and momentum indicators. | **Feature Richness.** Enhanced predictive power through technical analysis-based features. |
| **4. Data Splitting** | Split dataset into training and testing sets while maintaining temporal order (no data leakage). | **Model Validation.** Ensured realistic backtesting setup for time-series data. |

### Phase 2: Model Training and Evaluation

| Step | Aim | Achievement |
| :--- | :--- | :--- |
| **5. Model Selection** | Train multiple classification algorithms: **Logistic Regression**, **Support Vector Classifier**, and **XGBClassifier**. | **Algorithm Comparison.** Identified best-performing model through systematic evaluation. |
| **6. Hyperparameter Tuning** | Optimize model parameters using cross-validation techniques. | **Performance Optimization.** Fine-tuned models for maximum predictive accuracy. |
| **7. Model Evaluation** | Generate comprehensive performance metrics: accuracy, precision, recall, F1-score, and confusion matrices. | **Performance Insights.** Quantified model reliability and identified strengths/weaknesses. |
| **8. Visualization & Analysis** | Create performance plots, correlation heatmaps, and feature importance charts. | **Visual Storytelling.** Clear interpretation of model behavior and market patterns. |

---

## 📊 Model Performance Highlights

The notebook explores several classification models to predict Bitcoin price direction:

| Model | Accuracy | Precision (Up Move) | F1-Score |
| :--- | :--- | :--- | :--- |
| **🏆 XGBClassifier** | **~65-70%** | **Strong** | **Optimized** |
| Logistic Regression | ~55-60% | Moderate | Moderate |
| SVC | ~50-55% | Low | Low |

📈 **Key Insights**
* **Moving averages** prove to be strong predictive features.
* **Volume patterns** significantly enhance prediction accuracy.
* **Volatility measures** help identify market regime changes.

---

## ⚠️ Limitations & Real-World Considerations

This section addresses the inherent limitations of the model and practical challenges for real-world application:

### Predictive Limitations
* **Black Swan Events:** The model relies purely on **Technical Analysis** and cannot predict sudden market shocks (regulatory changes, exchange hacks, macroeconomic events).
* **Magnitude vs Direction:** Model predicts direction but not price magnitude, severely limiting profit/loss estimation capabilities.
* **Market Efficiency:** Bitcoin markets are becoming increasingly efficient, making consistent directional prediction challenging.

### Scalability and Infrastructure Constraints
* **Storage Requirements:** Transitioning to minute-level data would require **high-performance NVMe SSDs** or distributed storage solutions (AWS S3, Google Cloud Storage) due to exponential data growth.
* **Computational Power:** Training on high-frequency data demands significant resources—**powerful GPUs** or distributed computing frameworks like **Apache Spark**.
* **Memory Management:** Processing large datasets requires substantial **RAM** to avoid memory overflow and maintain optimal processing speeds.

---

## 🚀 Future Modifications & Enhancements

The following enhancements are planned to improve model robustness and transition toward production-ready deployment:

### 🔮 Advanced Modeling
* **Deep Learning Integration:** Implement **LSTM** or **Transformer** architectures for better sequential pattern recognition.
* **Multi-timeframe Analysis:** Incorporate predictions across different time horizons (hourly, daily, weekly).

### 📰 Alternative Data Sources
* **Sentiment Analysis:** Integrate social media feeds (Twitter/X, Reddit) and news headlines for market sentiment indicators.
* **On-chain Metrics:** Include blockchain metrics (hash rate, active addresses, transaction volume) for fundamental analysis.

### 🚀 Production Deployment
* **API Development:** Wrap trained models in **FastAPI** or **Flask** for real-time prediction serving.
* **MLOps Pipeline:** Implement automated model retraining, versioning, and monitoring using **MLflow** or **Kubeflow**.

---

## ⚙️ How to Run the Analysis

### 🔧 Prerequisites
* Python 3.8 or higher
* Jupyter Lab/Notebook or VS Code with Python extension

### 📥 Installation & Setup
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Aditya-git-rajya/bitcoin-price-prediction.git](https://github.com/Aditya-git-rajya/bitcoin-price-prediction.git)
    cd bitcoin-price-prediction
    ```
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv bitcoin_ml_env
    source bitcoin_ml_env/bin/activate  # On Windows: bitcoin_ml_env\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
    ```
4.  **Launch Analysis:**
    ```bash
    jupyter notebook Bitcoin-Price-Prediction-using-Machine-Learning-in-Python.ipynb
    ```

---

<div align="center">
    <img src="https://img.shields.io/badge/Made%20with-Python-FFD700?style=for-the-badge&logo=python" alt="Made with Python">
    <img src="https://img.shields.io/badge/Powered%20by-XGBoost-FF8C00?style=for-the-badge&logo=xgboost" alt="Powered by XGBoost">
    <img src="https://img.shields.io/badge/Built%20for-Crypto%20Trading-FFA500?style=for-the-badge&logo=bitcoin" alt="Built for Crypto">
</div>
<div align="center"> 
    <h3>⭐ Star this repository if you found it helpful! ⭐</h3> 
</div>
