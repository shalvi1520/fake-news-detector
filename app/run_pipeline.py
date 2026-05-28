"""
One-click pipeline: Preprocess → Train → Evaluate
Run: python run_pipeline.py
"""
import sys
sys.path.insert(0, ".")

print("\n" + "="*60)
print("  FAKE NEWS DETECTOR — FULL PIPELINE")
print("="*60)

# Step 1: Preprocess
print("\n[1/3] PREPROCESSING...")
from src.preprocess import run_preprocessing
run_preprocessing()

# Step 2: Train
print("\n[2/3] TRAINING MODELS...")
from src.train import run_training
run_training()

# Step 3: Evaluate
print("\n[3/3] EVALUATING...")
from src.evaluate import run_evaluation
run_evaluation()

print("\n" + "="*60)
print("  ✅ ALL DONE! Launch app with:")
print("  streamlit run app/app.py")
print("="*60 + "\n")