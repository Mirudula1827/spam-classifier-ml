Spam classification using Machine Learning with TF-IDF and Logistic Regression

## 📌 Overview

This project builds a machine learning model to classify text messages as **Spam** or **Not Spam** using Natural Language Processing techniques.

---

## ⚙️ Features

* Text preprocessing (lowercasing, stopword removal, lemmatization)
* TF-IDF vectorization
* Logistic Regression model
* Pipeline integration for end-to-end processing
* Evaluation using Accuracy, Precision, Recall, and F1 Score

---

## 🧠 Model Pipeline

Raw Text → Cleaning → TF-IDF → Logistic Regression → Prediction

---

## 📊 Performance

* Accuracy: ~97%
* Precision: ~91%
* Recall: ~91%
* F1 Score: ~91%

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook.

---

## 🧪 Example Predictions

| Message                                               | Prediction |
| ----------------------------                          | ---------- |
| "URGENT! You have won a 1lakh cash prize. Call now!"  | Spam       |
| "Hey, are we meeting today?"                          | Not Spam   |

---

## 🎯 Key Insight

The model balances precision and recall effectively, making it suitable for real-world spam detection scenarios.

---

## 📚 Technologies Used

* Python
* scikit-learn
* NLTK
* Pandas

---

## 👨‍💻 Author

Mirudula N D
