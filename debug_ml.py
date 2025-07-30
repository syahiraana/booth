# debug_ml.py - Step by step debugging
import sys
print("Step 1: Python path")
print(sys.path)

print("Step 2: Import pandas")
import pandas as pd
print("✅ Pandas imported")

print("Step 3: Import numpy") 
import numpy as np
print("✅ Numpy imported")

print("Step 4: Import sklearn")
from sklearn.ensemble import RandomForestClassifier
print("✅ Sklearn imported")

print("Step 5: Import Django modules")
import os
import django
print("✅ Django module imported")

print("Step 6: Set Django settings")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skillgapanalysis.settings')
print("✅ Django settings set")

print("Step 7: Django setup")
django.setup()
print("✅ Django setup completed")

print("Step 8: Import Django models")
try:
    from core.models import Student, MLModel
    print("✅ Models imported successfully")
except Exception as e:
    print(f"❌ Model import failed: {e}")

print("Step 9: Check CSV file")
if os.path.exists('job_preference_analysis_data.csv'):
    df = pd.read_csv('job_preference_analysis_data.csv')
    print(f"✅ CSV loaded: {df.shape}")
else:
    print("❌ CSV not found")

print("🎉 All debug steps completed!")
