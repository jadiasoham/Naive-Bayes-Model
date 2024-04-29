from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import string
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Function for cleaning and normalizing text:
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'^[a-zA-Z]\s', '', text)
    text = re.sub(r'\s+', '', text)
    return text

# A function to plot histogram
def plot_histogram(data, x_label, y_label, title, n_bins=None):
    if n_bins is None:
        n = len(data)
        q3, q1 = np.percentile(data, [75, 25])
        iqr = q3 - q1
        bin_width = 2 * iqr * (n**(-1/3))
        if np.isinf(bin_width) or np.isnan(bin_width) or bin_width == 0:
            n_bins = np.ceil(2*(np.cbrt(n))).astype('int32')
        else:
            data_range = max(data) - min(data)
            n_bins = np.ceil(data_range/bin_width).astype('int32')
    plt.hist(x=data, bins=n_bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Function to plot heatmap
def plot_heatmap(data, x_label, y_label, x_cat, y_cat, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', xticklabels=x_cat, yticklabels=y_cat)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()

# Load the dataset
data_file = 'user_courses_review_test_set.csv'
data = pd.read_csv(data_file)

# Data cleaning and exploration
data['review_rating'] = data['review_rating'].astype(np.float64)
cleaned_data = data.dropna()
cleaned_data['comment_length'] = cleaned_data['review_comment'].apply(len)

# Text processing
cleaned_data['review_comment'] = cleaned_data['review_comment'].apply(clean_text)
X = cleaned_data['review_comment']
y = cleaned_data['review_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building and evaluation
vectorizer = TfidfVectorizer(stop_words='english')
model = MultinomialNB()
text_clf = make_pipeline(vectorizer, model)
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)
report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# Addressing class imbalance
df_majority = cleaned_data[cleaned_data.review_rating > 4]
df_minority = cleaned_data[cleaned_data.review_rating <= 4]
df_minority_sampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_unsampled = pd.concat([df_majority, df_minority_sampled])
X_unsampled = df_unsampled['review_comment']
y_unsampled = df_unsampled['review_rating']
X_us_train, X_us_test, y_us_train, y_us_test = train_test_split(X_unsampled, y_unsampled, test_size=0.2, random_state=42)
text_clf_unsamp = make_pipeline(vectorizer, model)
text_clf_unsamp.fit(X_us_train, y_us_train)
y_us_pred = text_clf_unsamp.predict(X_us_test)
report_unsampled = classification_report(y_us_test, y_us_pred)
conf_mat_unsampled = confusion_matrix(y_us_test, y_us_test)

# Binary classification
bin_data = cleaned_data.copy()
bin_data['review_rating'] = bin_data['review_rating'].apply(lambda x: 'good' if x >= 4 else 'bad')
df_majority_bin = bin_data[bin_data.review_rating == 'good']
df_minority_bin = bin_data[bin_data.review_rating == 'bad']
df_minority_unsampled_bin = resample(df_minority_bin, replace=True, n_samples=len(df_majority_bin), random_state=42)
df_unsampled_bin = pd.concat([df_majority_bin, df_minority_unsampled_bin])
X_unsmp_bin = df_unsampled_bin['review_comment']
y_unsmp_bin = df_unsampled_bin['review_rating']
X_unsmp_bin_train, X_unsmp_bin_test, y_unsmp_bin_train, y_unsmp_bin_test = train_test_split(X_unsmp_bin, y_unsmp_bin, test_size=0.2, random_state=42)
clf_txt_bin_unsmp = make_pipeline(vectorizer, BernoulliNB())
clf_txt_bin_unsmp.fit(X_unsmp_bin_train, y_unsmp_bin_train)
y_pred_bin_unsmp = clf_txt_bin_unsmp.predict(X_unsmp_bin_test)
bin_report_unsmp = classification_report(y_unsmp_bin_test, y_pred_bin_unsmp, output_dict=True)
conf_mat_unsamp_bin = confusion_matrix(y_unsmp_bin_test, y_pred_bin_unsmp)

# Plotting
plot_heatmap(conf_mat_unsampled, 'Predicted', 'Actual', x_cat=['2', '3', '4', '5'], y_cat=['2', '3', '4', '5'], title='Confusion Matrix for unsampled data')
plot_heatmap(conf_mat_unsamp_bin, 'Predicted', 'Actual', ['Bad', 'Good'], ['Bad', 'Good'], 'Confusion Matrix For Binary Classification')
