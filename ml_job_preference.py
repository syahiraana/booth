# ml_job_preference.py - MEMORY EFFICIENT FOR 90-97% ACCURACY
import pandas as pd
import numpy as np
import joblib
import os
import sys
import django
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("🔧 Setting up Django...")
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skillgapanalysis.settings')
    django.setup()
    from core.models import MLModel
    print("✅ Django setup successful")
except Exception as e:
    print(f"⚠️ Django setup warning: {e}")

class MemoryEfficientMLTrainer:
    def __init__(self, data_file='job_preference_analysis_data.csv'):
        self.data_file = data_file
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.model_dir = 'ml_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess balanced data"""
        print("📊 Loading balanced dataset...")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"❌ Run ETL script first: python etl_job_preference_analyze.py")
        
        df = pd.read_csv(self.data_file)
        print(f"📈 Dataset shape: {df.shape}")
        
        # Prepare features
        feature_columns = [
            'possessed_skills_count', 'missing_skills_count', 'skills_match_ratio',
            'certificates_count', 'certificate_ratio', 'grades_score', 'gpa_value',
            'total_skills', 'certified_skills', 'required_skills_count'
        ]
        
        X = df[feature_columns].fillna(0)
        y = df['engagement_label']
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"🎯 Features: {list(X.columns)}")
        print(f"📊 Target distribution:")
        for label, count in pd.Series(y).value_counts().items():
            print(f"   {label}: {count} ({count/len(y)*100:.1f}%)")
        
        self.feature_names = list(X.columns)
        return X, y_encoded, df
    
    def train_memory_efficient_model(self, X_train, y_train):
        """Train memory-efficient model for 90-97% accuracy"""
        print("🤖 Training memory-efficient RandomForest...")
        
        # Memory-efficient parameters (no GridSearch to avoid memory issues)
        self.model = RandomForestClassifier(
            n_estimators=150,          # Moderate number
            max_depth=10,              # Controlled depth
            min_samples_split=4,       # Conservative splitting
            min_samples_leaf=2,        # Prevent overfitting
            max_features='sqrt',       # Feature randomness
            bootstrap=True,
            random_state=42,
            class_weight='balanced',   # Handle any remaining imbalance
            max_samples=0.9,           # Bootstrap sampling
            oob_score=True,            # Out-of-bag validation
            n_jobs=2                   # Limit parallel jobs to save memory
        )
        
        self.model.fit(X_train, y_train)
        print("✅ Model training completed")
        
        if hasattr(self.model, 'oob_score_'):
            print(f"📊 Out-of-bag score: {self.model.oob_score_:.4f}")
        
        return self.model
    
    def evaluate_model_for_target_range(self, X_train, X_test, y_train, y_test):
        """Evaluate model with focus on 90-97% target"""
        print("📋 Evaluating model performance...")
        print("=" * 50)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"📊 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"🔍 Precision: {precision:.4f}")
        print(f"📈 Recall: {recall:.4f}")
        print(f"⚖️ F1-Score: {f1:.4f}")
        
        # ✅ TARGET ACCURACY CHECK (90-97%)
        print(f"\n🎯 ACCURACY TARGET ANALYSIS:")
        if 0.90 <= test_accuracy <= 0.97:
            print(f"🎉 PERFECT: Test accuracy {test_accuracy:.4f} is within target range (90-97%)")
            print("🏆 Model meets EXACT requirements!")
            status = "PERFECT"
        elif test_accuracy > 0.97:
            print(f"⚠️ TOO HIGH: Test accuracy {test_accuracy:.4f} above 97%")
            print("💡 Possible overfitting")
            status = "HIGH"
        elif 0.85 <= test_accuracy < 0.90:
            print(f"⚠️ CLOSE: Test accuracy {test_accuracy:.4f} close to target")
            print("💡 Need minor tuning")
            status = "CLOSE"
        else:
            print(f"❌ MISS: Test accuracy {test_accuracy:.4f} outside target")
            status = "MISS"
        
        # Cross-validation (memory efficient - only 3 folds)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=3, scoring='accuracy')
        print(f"\n🔄 Cross-Validation:")
        print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # ✅ DETAILED CLASSIFICATION REPORT
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print("=" * 50)
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_test_pred, target_names=target_names, digits=4)
        print(report)
        
        # Confusion Matrix
        print(f"\n📊 CONFUSION MATRIX:")
        cm = confusion_matrix(y_test, y_test_pred)
        if len(target_names) == 2:
            print(f"           Predicted")
            print(f"         {target_names[0]:>4s}  {target_names[1]:>4s}")
            print(f"Actual {target_names[0]:>3s}  {cm[0,0]:3d}   {cm[0,1]:3d}")
            print(f"      {target_names[1]:>4s}  {cm[1,0]:3d}   {cm[1,1]:3d}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n🔍 FEATURE IMPORTANCE:")
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']:25s}: {row['importance']:.4f}")
        
        print("=" * 50)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'target_met': 0.90 <= test_accuracy <= 0.97,
            'status': status
        }
    
    def save_models(self, metrics):
        """Save models and metadata"""
        print("💾 Saving models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.model_dir, f'job_preference_model_{timestamp}.pkl')
        scaler_path = os.path.join(self.model_dir, f'job_preference_scaler_{timestamp}.pkl')
        encoder_path = os.path.join(self.model_dir, f'job_preference_encoder_{timestamp}.pkl')  # ✅ ADD THIS
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)  # ✅ SAVE ENCODER
        
        print(f"📁 Models saved:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Encoder: {encoder_path}")  # ✅ ADD THIS
        
        # Save to database
        try:
            MLModel.objects.filter(model_name='job_preference_analyzer', is_active=True).update(is_active=False)
            
            MLModel.objects.create(
                model_name='job_preference_analyzer',
                model_version=timestamp,
                model_type='RandomForestClassifier',
                accuracy=metrics['test_accuracy'],
                precision_score=metrics['precision'],
                recall_score=metrics['recall'],
                f1_score=metrics['f1_score'],
                model_file_path=model_path,
                scaler_file_path=scaler_path,
                pca_file_path=encoder_path,  # ✅ KEEP FOR BACKWARD COMPATIBILITY
                encoder_file_path=encoder_path,  # ✅ ADD NEW FIELD
                training_samples=400,
                feature_count=len(self.feature_names),
                is_active=True
            )
            print("✅ Saved to database")
        except Exception as e:
            print(f"⚠️ Database save failed: {e}")
        
        return {'timestamp': timestamp, 'model_path': model_path}

    
    def train_pipeline(self):
        """Complete memory-efficient training pipeline"""
        print("🚀 Starting MEMORY-EFFICIENT ML Training Pipeline...")
        print("🎯 TARGET: 90-97% Test Accuracy")
        print("=" * 70)
        
        try:
            # Load data
            X, y, df = self.load_and_preprocess_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Training: {len(X_train)} samples")
            print(f"📊 Testing: {len(X_test)} samples")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (memory efficient)
            self.train_memory_efficient_model(X_train_scaled, y_train)
            
            # Evaluate
            metrics = self.evaluate_model_for_target_range(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Save
            paths = self.save_models(metrics)
            
            print("=" * 70)
            print("🎉 TRAINING COMPLETED!")
            print(f"🎯 Final Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
            
            if metrics['target_met']:
                print("✅ SUCCESS: Target 90-97% accuracy achieved!")
            else:
                print(f"⚠️ Status: {metrics['status']}")
            
            return metrics, paths
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

if __name__ == "__main__":
    if not os.path.exists('job_preference_analysis_data.csv'):
        print("❌ Run ETL first: python etl_job_preference_analyze.py")
        sys.exit(1)
    
    trainer = MemoryEfficientMLTrainer()
    metrics, paths = trainer.train_pipeline()
    
    if metrics:
        print(f"\n🏆 FINAL RESULT:")
        print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   Target Met: {'YES ✅' if metrics['target_met'] else 'NO ❌'}")
        print(f"   Status: {metrics['status']}")
