# 📧 Spam Classifier using NLP & Logistic Regression

A machine learning web application that classifies text messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a Logistic Regression model.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Deployed](https://img.shields.io/badge/Live-Streamlit-red)

---

## 📌 Overview

This project implements an end-to-end text classification pipeline to detect spam messages. It uses **TF-IDF vectorization** combined with a **Logistic Regression classifier** to achieve high accuracy and real-time predictions.

The application is deployed using **Streamlit**, providing an interactive interface where users can input messages and instantly receive predictions with confidence scores.


---

## 🚀 Live Demo

👉 [Click here to try the app](https://spam-classifier-ml-erogxrj8mfzzmyeohcha5r.streamlit.app/)

---

## 🚀 Highlights

- 🧠 NLP preprocessing (tokenization, stopword removal, lemmatization)
- 🔤 TF-IDF vectorization for feature extraction
- 🤖 Logistic Regression model with class balancing
- ⚡ Real-time predictions with confidence scores
- 🎯 Supports multiple message inputs (one per line)
- 🎨 Clean and interactive UI built with Streamlit
- 📊 Model performance metrics displayed in-app


---

## 📊 Model Performance

- Accuracy: ~97.5%
- Precision: ~91%
- Recall: ~91%
- F1 Score: ~91%

---

## 🔁 Workflow Overview

| Step | Description |
|------|------------|
| 🔍 Data Loading | Import SMS dataset |
| 🧹 Preprocessing | Clean text, remove stopwords, lemmatize |
| 🔤 Vectorization | Convert text to numerical form using TF-IDF |
| 🧠 Model Training | Train Logistic Regression classifier |
| 📈 Evaluation | Compute Accuracy, Precision, Recall, F1 Score |
| 🌐 Deployment | Build UI and deploy using Streamlit |

---
**Dataset**: SMS Spam Collection Dataset(Kaggle)

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

---

## ▶️ Run Locally

```bash
git clone https://github.com/Mirudula1827/spam-classifier-ml.git
cd spam-classifier-ml
pip install -r requirements.txt
streamlit run app.py

```
---
## 📊 Sample Output

Input:
Win money now!!!

Output:
🚨 Spam (92%)

---

## ⚠️ Limitations

- 📉 Model trained on SMS dataset; may not generalize to all domains
- 🌍 Does not account for evolving spam patterns
- 🧠 Limited contextual understanding compared to deep learning models
---

## 💡 Future Improvements
- Use deep learning models (LSTM / Transformers)
- Deploy as REST API using Flask/FastAPI
- Add multilingual support
- Integrate real-time spam detection for emails/messages

---

## 📄 License

This project is licensed under the MIT License.
---
---
## 👨‍💻 Author

Mirudula N D
---
