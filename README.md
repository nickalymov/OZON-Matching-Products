# OZON-Matching-Products
## Ozon Tech ML Competition: Product Matching
This repository contains a solution to the product matching task based on product names, attributes, and images. The work was done as part of the Ozon Tech competition and achieved 7th place out of 110 teams with a ROC-AUC score of 0.9216.

## 🚀 Task
Develop a machine learning model that, given information about two products (text descriptions, images, attributes), predicts whether they are the same (target = 1) or not (target = 0).

## 📦 Data
The organizers provided the following datasets:

train.csv – pairs of products with a label (variantid1, variantid2, target)

test.csv – same format, but without labels

attributes.csv – product categories and attributes

text.csv – titles, descriptions, and BERT embeddings

resnet.csv – ResNet embeddings of product images

## 🔧 Data Processing
Processing includes basic cleaning and feature generation based on:

Textual data: Levenshtein distance, Jaccard similarity, string length, etc.

Categories: matches and subcategories

Attributes: Jaccard similarity, difference in counts, binary flags

Image embeddings: cosine and Euclidean distances, entropy

For more details, see the notebooks:

1_main_features.ipynb

2_add_cat_features.ipynb

## 🧠 Models
### 1. AutoGluon Tabular (0.9216)
A multi-model stack with automatic ensembling. Presets used:

best_quality — for maximum accuracy

zeroshot — for faster experimentation

Files:

autogluon.ipynb – training and hyperparameter tuning

ag_inference.ipynb – loading model predictions

📸 Models used within AutoGluon:
![image](https://github.com/user-attachments/assets/02dd26a5-79e8-48d6-b932-ed624a16e689)


### 2. HistGradientBoosting + Optuna (0.91)
An alternative lightweight pipeline using HistGradientBoostingClassifier from sklearn, optimized with Optuna.

Handles class imbalance well

Minimal inference time

Ready for production integration

Files:

training_hgb.ipynb – training and hyperparameter tuning

hgb_inference.ipynb – full pipeline: from raw data to predictions

## 🏁 Results
AutoGluon (stacked): ROC-AUC 0.9216 – top 7 on the leaderboard
HistGradientBoosting + Optuna: faster, slightly lower score, but more production-friendly

## 🔍 Analysis
The solution achieved a high result due to strong feature engineering and AutoML. However, teams ranked higher in the leaderboard utilized pre-trained language models (e.g., BERT) for processing categories and attributes, which led to an improvement in quality.
