import os, json, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix
from joblib import dump

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "reviews.csv"))

# Target: low-star (<=2)
df = df.dropna(subset=["content"])
df["low_star"] = (df["rating"] <= 2).astype(int)

# Basic text cleanup (keep as-is; TF-IDF will handle)
X = df[["content","brand"]]; y=df["low_star"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("text", TfidfVectorizer(min_df=5, ngram_range=(1,2)), "content"),
    ("brand", OneHotEncoder(handle_unknown="ignore"), ["brand"]),
])

logit = Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=2000, class_weight='balanced'))])
rf = Pipeline([("pre",pre),("clf",RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced'))])

metrics = {}
outputs = {}

for name, pipe in {"logit":logit,"rf":rf}.items():
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    preds = (proba>=0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    p,r,f1,_ = precision_recall_fscore_support(yte, preds, average="binary", zero_division=0)
    metrics[name] = {"AUC":round(float(auc),3),"precision":round(float(p),3),"recall":round(float(r),3),"f1":round(float(f1),3)}
    dump(pipe, os.path.join(ROOT,"models",f"{name}.joblib"))

# Top features (logit)
pipe = logit; pipe.fit(Xtr, ytr)
vec = pipe.named_steps["pre"].named_transformers_["text"]
enc = pipe.named_steps["pre"].named_transformers_["brand"]
names = list(vec.get_feature_names_out()) + list(enc.get_feature_names_out(["brand"]))
coefs = pipe.named_steps["clf"].coef_[0]
(pd.DataFrame({"feature":names, "coef":coefs})
   .sort_values("coef", ascending=False).head(40)
   .to_csv(os.path.join(ROOT,"reports","logit_top_coefs.csv"), index=False))

# RF importances
pipe = rf; pipe.fit(Xtr, ytr)
vec = pipe.named_steps["pre"].named_transformers_["text"]
enc = pipe.named_steps["pre"].named_transformers_["brand"]
names = list(vec.get_feature_names_out()) + list(enc.get_feature_names_out(["brand"]))
(pd.DataFrame({"feature":names, "importance":pipe.named_steps["clf"].feature_importances_})
   .sort_values("importance", ascending=False).head(40)
   .to_csv(os.path.join(ROOT,"reports","rf_feature_importances.csv"), index=False))

os.makedirs(os.path.join(ROOT,"reports"), exist_ok=True)
with open(os.path.join(ROOT,"reports","metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Wrote metrics to reports/metrics.json")
