# Employee Review Sentiment Analysis & Topic Modeling

## Overview

This project applies data mining and natural language processing (NLP) techniques to analyze employee reviews and extract meaningful insights from unstructured text data.

Using employee reviews from Glassdoor, the project combines exploratory data analysis, sentiment analysis, topic modeling, and machine learning to identify key workplace themes and predict review sentiment.

The goal is to demonstrate how organizations can transform large volumes of textual feedback into actionable insights that support better decision-making and employee satisfaction initiatives.

---

## Dataset

The dataset consists of employee reviews collected from Glassdoor. 
Download it from : https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews

Each review contains:

* Employee ratings
* Pros and cons text
* Additional review metadata

For this project:

* Review text was used for text mining and sentiment analysis.
* Ratings were used to create sentiment labels for supervised learning.

---

## Project Workflow

### 1. Data Preprocessing

Text preprocessing was performed to prepare reviews for analysis:

* Text cleaning
* Lowercasing
* Stopword removal
* Tokenization
* Lemmatization
* Creation of a combined review text field

---

### 2. Exploratory Data Analysis (EDA)

EDA was conducted to understand the structure and characteristics of the dataset.

Key findings:

* Reviews exhibit a strong positive sentiment bias.
* Sentiment scores are concentrated toward the positive end of the scale.
* Frequently discussed workplace topics include:

  * Management
  * Salary
  * Work-life balance
  * Career opportunities
  * Employee benefits

---

### 3. Sentiment Analysis

VADER (Valence Aware Dictionary and Sentiment Reasoner) was used to calculate sentiment scores for each review.

Key statistics:

* Number of reviews analyzed: 838,551
* Mean sentiment score: 0.571
* Median sentiment score: 0.710
* Sentiment range: -0.998 to 0.999

The results indicate an overall positive perception of workplaces while still capturing meaningful negative feedback.

---

### 4. Topic Modeling

Latent Dirichlet Allocation (LDA) was applied to uncover hidden themes within employee reviews.

Major topics identified:

* Management and leadership
* Compensation and salary
* Work-life balance
* Career growth
* Benefits and workplace environment

These topics represent the primary concerns discussed by employees.

---

### 5. Sentiment Classification

A supervised machine learning pipeline was built using:

* TF-IDF Vectorization
* Logistic Regression

The model was trained to classify reviews as positive or negative.

#### Feature Engineering

```python
tfidf = TfidfVectorizer(max_features=5000)
```

#### Classification Model

```python
LogisticRegression(class_weight="balanced")
```

The use of class balancing helped improve detection of minority (negative) reviews.

---

### 6. Model Evaluation

Performance was assessed using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross-Validation

Results showed:

* Strong overall classification performance
* High effectiveness on positive reviews
* Reasonable performance on negative reviews despite class imbalance
* Stable results across validation folds

Cross-validation achieved:

* F1 Score: 0.884 ± 0.010

---

### 7. Hybrid Sentiment System

To improve robustness, a hybrid approach was implemented by combining:

* Logistic Regression predictions
* VADER sentiment analysis

This approach improves handling of mixed or ambiguous reviews and produces more balanced sentiment predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* Gensim
* VADER Sentiment Analysis

---

## Key Insights

* Employee reviews are predominantly positive.
* Management, salary, and work-life balance are the most frequently discussed workplace topics.
* Sentiment analysis can effectively quantify employee satisfaction.
* Machine learning models can accurately classify employee sentiment from textual reviews.
* Topic modeling provides actionable insights into organizational strengths and weaknesses.

---

## Future Improvements

Potential enhancements include:

* Multi-class sentiment classification
* Deep learning approaches (LSTM, BERT)
* BERTopic for advanced topic modeling
* Interactive dashboards for HR analytics
* Real-time employee feedback monitoring

---

## Author

**Mohamed Aziz Amri /** 
**Mohamed Ben Arbia /**
**Rissen Guermazi**

Bachelor of Business Administration
Major: Business Analytics
Minor: Information Technology

Tunis Business School

