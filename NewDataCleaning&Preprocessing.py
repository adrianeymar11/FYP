import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
DATA_PATH = "digital_wellbeing_dataset.csv"
assert os.path.exists(DATA_PATH), f"File not found at {DATA_PATH}. Upload dataset to this path."

df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# -------------------------------
# 2. Target detection / binning
# -------------------------------
target_col = "Mental_Health_Score"
if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
    print(f"Target '{target_col}' is numeric — binning into 3 categories.")
    df[target_col + "_binned"] = pd.qcut(df[target_col], 3, labels=['Low', 'Medium', 'High'])
    target_col = target_col + "_binned"

print("Final target column:", target_col)
print(df[target_col].value_counts())

# -------------------------------
# 3. Feature selection
# -------------------------------
excluded = [target_col]
X = df.drop(columns=excluded)
y = df[target_col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# -------------------------------
# 4. Preprocessing pipelines
# -------------------------------
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -------------------------------
# 5. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# -------------------------------
# 6. Train 4 models for comparison
# -------------------------------
models = {
    'LogisticRegression': (LogisticRegression(max_iter=2000, solver='liblinear', multi_class='ovr'),
                            {'clf__C': [0.01, 0.1, 1, 10]}),
    'KNeighbors': (KNeighborsClassifier(),
                   {'clf__n_neighbors': [3, 5, 7], 'clf__weights': ['uniform', 'distance']}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42),
                     {'clf__max_depth': [3, 5, 8, None], 'clf__min_samples_split': [2, 5, 10]}),
    'RandomForest': (RandomForestClassifier(random_state=42, n_jobs=-1),
                     {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10, None], 'clf__min_samples_split': [2, 5]})
}

results = {}
for name, (clf, grid) in models.items():
    print(f"\n==== Training {name} ====")
    pipe = ImbPipeline(steps=[('preproc', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', clf)])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Best params: {gs.best_params_}")
    print(f"Test accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(best, f"{name}_best_pipeline.pkl")
    results[name] = {'test_acc': acc, 'model': best, 'y_pred': y_pred}

# -------------------------------
# 7. Comparison plot
# -------------------------------
res_df = pd.DataFrame([{'model': name, 'test_acc': info['test_acc']} for name, info in results.items()]).sort_values('test_acc', ascending=False)
print("\nModel comparison:")
print(res_df)
plt.figure(figsize=(8, 4))
sns.barplot(data=res_df, x='model', y='test_acc')
plt.ylim(0, 1)
plt.title("Test Accuracy by Model")
plt.show()

# -------------------------------
# 8. Confusion matrix for best model
# -------------------------------
best_name = res_df.iloc[0]['model']
best_model = results[best_name]['model']
y_pred_best = results[best_name]['y_pred']
cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - {best_name}")
plt.show()

# -------------------------------
# 9. Train simplified dashboard model
# -------------------------------
print("\n🔁 Training simplified model for user dashboard...")
selected_features = [
    'Age', 'Sleep_Hours', 'Daily_Screen_Time_Hours', 'Gaming_Usage_Hours',
    'Social_Media_Usage_Hours', 'Stress_Level', 'Physical_Activity_Hours',
    'Support_Systems_Access', 'Online_Support_Usage', 'Work_Environment_Impact'
]
selected_features = [f for f in selected_features if f in df.columns]
print("Selected features available for retraining:", selected_features)

X_simplified = df[selected_features].copy()
y_simplified = y.copy()
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simplified, y_simplified, test_size=0.2, random_state=42, stratify=y_simplified)

num_features_s = X_simplified.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features_s = X_simplified.select_dtypes(include=['object', 'category']).columns.tolist()

num_tf_s = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_tf_s = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preproc_s = ColumnTransformer([('num', num_tf_s, num_features_s), ('cat', cat_tf_s, cat_features_s)])

rf_simple = RandomForestClassifier(n_estimators=200, random_state=42)
pipe_simple = Pipeline([('preproc', preproc_s), ('clf', rf_simple)])
pipe_simple.fit(X_train_s, y_train_s)
print(f"Simplified dashboard model accuracy: {accuracy_score(y_test_s, pipe_simple.predict(X_test_s)):.3f}")
print("Classification report:\n", classification_report(y_test_s, pipe_simple.predict(X_test_s)))

joblib.dump(pipe_simple, "RandomForest_user_pipeline.joblib")
print("✅ Saved simplified model to RandomForest_user_pipeline.joblib")

# -------------------------------
# 10. Impute missing values globally before saving cleaned dataset
# -------------------------------
for col in ['Sleep_Hours', 'Daily_Screen_Time_Hours', 'Gaming_Usage_Hours', 'Social_Media_Usage_Hours', 'Physical_Activity_Hours']:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

for col, default in [
    ('Stress_Level', 'Medium'),
    ('mental_health_consequence', 'Mild'),
    ('Self_Reported_Addiction_Level', 'Moderate'),
    ('phys_health_consequence', 'Moderate')
]:
    if col in df.columns:
        df[col].fillna(default, inplace=True)

print("\nMissing values after imputation:")
print(df.isna().sum().sort_values(ascending=False).head(10))

df.to_csv("digital_wellbeing_cleaned.csv", index=False)
print("✅ Cleaned dataset saved → digital_wellbeing_cleaned.csv")