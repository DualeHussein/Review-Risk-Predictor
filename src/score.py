import sys, os
import pandas as pd
from joblib import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_p = os.path.join(ROOT,"models","logit.joblib")
if not os.path.exists(model_p):
    print("Model not found. Train first: python src/train.py"); sys.exit(1)
pipe = load(model_p)

text = " ".join(sys.argv[1:]).strip()
if not text:
    print('Usage: python src/score.py "your review text"'); sys.exit(1)

X = pd.DataFrame([{"content":text, "brand":"JBL"}])
proba = pipe.predict_proba(X)[:,1][0]
print(f"Low-star risk: {proba:.3f}")
