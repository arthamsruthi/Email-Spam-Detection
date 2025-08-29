# Email-Spam-Detection
The goal of this project is to **train a spam detection model** that can accurately distinguish between spam and non-spam emails, improving communication safety and reducing the chances of falling victim to phishing or scams.

## 📌 Overview

Email spam, or junk mail, is a common problem in digital communication. Spam emails often contain phishing attempts, scams, and advertisements, which can be harmful to users. This project uses **Machine Learning** to build an **Email Spam Classifier** that detects whether an email is **Spam 🚨** or **Not Spam ✅**.

## 🎯 Objectives

* Preprocess raw email text data (cleaning, removing punctuation, etc.).
* Convert text into numerical features using **TF-IDF Vectorization**.
* Train and evaluate different ML models for spam detection.
* Compare performance of models and choose the best one.
* Allow users to test custom emails with the trained model.

## 🛠️ Technologies Used

* **Python**
* **Pandas, NumPy** – Data handling
* **Scikit-learn** – Machine learning algorithms & evaluation
* **Matplotlib, Seaborn** – Data visualization

## 📂 Dataset

The project uses the **SMS Spam Collection Dataset** which contains labeled messages as `ham` (not spam) or `spam`.
You can download it from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## 🚀 Implementation Steps

1. **Load Dataset** – Import and clean the dataset.
2. **Preprocess Data** – Convert text to lowercase, remove numbers/punctuation.
3. **Feature Extraction** – Use **TF-IDF Vectorizer** to transform text into numerical form.
4. **Train Models** – Apply different ML algorithms:

   * Naive Bayes
   * Logistic Regression
   * Random Forest
   * Support Vector Machine (SVM)
5. **Evaluate Models** – Compare accuracy, confusion matrix, and classification report.
6. **Custom Prediction** – Test the model with new user-input messages.

## 📊 Model Performance

* The models achieved high accuracy (above 95%).
* **Naive Bayes** performed best for spam detection due to its suitability for text classification tasks.

## 🧪 Example Predictions

```
Input: "Congratulations! You won a free lottery of $1,000,000"
Output: Spam 🚨

Input: "Hi, let's meet tomorrow for the project discussion."
Output: Not Spam ✅

Input: "URGENT! Update your bank details immediately to avoid suspension."
Output: Spam 🚨
```

## 📸 Visualizations

* Distribution of Spam vs Ham messages
* Accuracy comparison of models (bar chart)

## 📌 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/email-spam-detection.git
   cd email-spam-detection
   ```
2. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter Notebook or Python script:

   ```bash
   jupyter notebook spam_detection.ipynb
   ```

   or

   ```bash
   python spam_detection.py
   ```

## 👨‍💻 Author

* **arthamsruthi**
