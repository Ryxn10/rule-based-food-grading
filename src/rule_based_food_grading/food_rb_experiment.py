import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'global_food_and_nutrition_database_2026.csv')

# Seed for deterministic execution
np.random.seed(42)

#Importing the dataset and preprocessing
def load_and_preprocess_real_data(filepath):
    
    #Loads the dataset from kaggle and preprocesses it to be used in the experiments.
    df_raw = pd.read_csv(filepath)
    
    # Standardise column names
    df = df_raw.rename(columns={
        'Energy_kcal': 'energy_kcal',
        'Sugar_g': 'sugar_g',
        'Saturated_Fat_g': 'sat_fat_g',
        'Sodium_mg': 'sodium_mg',
        'Fiber_g': 'fibre_g',
        'Protein_g': 'protein_g'
    })
    
    features = ['energy_kcal', 'sugar_g', 'sat_fat_g', 'sodium_mg', 'fibre_g', 'protein_g']
    df = df.dropna(subset=features)
    df = df[(df[features] >= 0).all(axis=1)]

    # Calculate underlying baseline score to serve as ground truth target label
    score = (df['sugar_g'] * 0.4) + (df['sat_fat_g'] * 0.8) + (df['sodium_mg'] * 0.005) + (df['energy_kcal'] * 0.01) - (df['fibre_g'] * 1.2) - (df['protein_g'] * 0.8)
    
    def assign_grade(s):
        if s < 5: return 'A'
        elif s < 12: return 'B'
        elif s < 20: return 'C'
        elif s < 30: return 'D'
        else: return 'E'
        
    df['nutri_grade'] = score.apply(assign_grade)
    return df

# Load data using absolute path
df = load_and_preprocess_real_data(CSV_PATH)

X = df[['energy_kcal', 'sugar_g', 'sat_fat_g', 'sodium_mg', 'fibre_g', 'protein_g']]
y = df['nutri_grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#Building algorithm 1
def rb_knowledge_engineered(row):
    """
    Expert system based on UK FSA thresholds.
    """
    neg_points = 0
    pos_points = 0

    if row['energy_kcal'] > 335: neg_points += 4
    elif row['energy_kcal'] > 167: neg_points += 2

    if row['sugar_g'] > 13.5: neg_points += 4
    elif row['sugar_g'] > 4.5: neg_points += 2

    if row['sat_fat_g'] > 5.0: neg_points += 4
    elif row['sat_fat_g'] > 1.5: neg_points += 2

    if row['sodium_mg'] > 900: neg_points += 4
    elif row['sodium_mg'] > 300: neg_points += 2

    if row['fibre_g'] > 4.7: pos_points += 4
    elif row['fibre_g'] > 1.9: pos_points += 2

    if row['protein_g'] > 6.4: pos_points += 4
    elif row['protein_g'] > 3.2: pos_points += 2

    net_score = neg_points - pos_points

    if net_score <= -1: return 'A'
    elif net_score <= 2: return 'B'
    elif net_score <= 8: return 'C'
    elif net_score <= 12: return 'D'
    else: return 'E'

y_pred_ke = X_test.apply(rb_knowledge_engineered, axis=1)

#Building algorithm 2

dt_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dd = dt_model.predict(X_test)

tree_rules = export_text(dt_model, feature_names=list(X.columns), max_depth=3)

#Experimental Evaluation
def compute_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    print(f"[{name}]")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    return acc, p, r, f1

print("--- EXPERIMENTAL EVALUATION RESULTS ---")
compute_metrics(y_test, y_pred_ke, "Algorithm 1: RB-KE (Expert System)")
compute_metrics(y_test, y_pred_dd, "Algorithm 2: RB-DD (Data-Driven)")

print("--- EXTRACTED IF-THEN DECISION TREE RULES (TOP 3 LEVELS) ---")
print(tree_rules)