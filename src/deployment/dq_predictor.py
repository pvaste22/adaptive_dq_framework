
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from common.logger import get_phase4_logger
from common.constants import PATHS

logger = get_phase4_logger('dq_predictor')

class DQScorePredictor:
    """Wrapper for trained ML model to predict DQ scores"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize predictor with trained model
        
        Args:
            model_dir: Path to model directory (auto-detects latest if None)
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        # Auto-detect latest model
        if model_dir is None:
            models_root = PATHS['artifacts'] / 'models'
            latest_link = models_root / 'latest'
            
            if latest_link.exists():
                # Read symlink target
                if latest_link.is_symlink():
                    model_dir = latest_link.resolve()
                elif latest_link.is_dir():
                    run_id_file = latest_link / "run_id.txt"
                    if run_id_file.exists():
                        with open(run_id_file, 'r') as f:
                            model_id = f.read().strip()
                        model_dir = models_root / model_id
                    else:
                        model_dir = latest_link
            else:
                with open(latest_link, 'r') as f:
                    rel_path = f.read().strip()
                model_dir = models_root / rel_path
            #else:
                #raise FileNotFoundError("No model found. Train model first!")
        
        self.model_dir = Path(model_dir)
        logger.info(f"Loading model from: {self.model_dir}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and metadata"""
        try:
            # Load model (prefer joblib over binary)
            model_pkl = self.model_dir / "model.joblib"
            model_bin = self.model_dir / "model.bin"
            
            if model_pkl.exists():
                self.model = joblib.load(model_pkl)
                logger.info("✓ Loaded model from joblib")
            elif model_bin.exists():
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(str(model_bin))
                logger.info("✓ Loaded model from binary")
            else:
                raise FileNotFoundError("No model file found")
            
            # Load scaler (optional)
            scaler_path = self.model_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✓ Loaded scaler")
            
            # Load metadata
            meta_path = self.model_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('features', [])
                logger.info(f"✓ Loaded metadata ({len(self.feature_names)} features)")
            else:
                logger.warning("No metadata found - feature order may be incorrect!")
            
            logger.info(f"Model loaded successfully (version: {self.metadata.get('version', 'unknown')})")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict DQ score for a single window
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Predicted DQ score (0.0 - 1.0)
        """
        try:
            # Align features with training order
            if self.feature_names:
                # Ensure all required features present
                feature_values = []
                for feat_name in self.feature_names:
                    if feat_name in features:
                        feature_values.append(features[feat_name])
                    else:
                        # Missing feature - use 0.0 or NaN
                        feature_values.append(0.0)
                        logger.debug(f"Missing feature: {feat_name}, using 0.0")
                
                X = np.array([feature_values])
            else:
                # Fallback: use features dict order
                X = np.array([list(features.values())])
            
            # Apply scaling if available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Predict
            #if hasattr(self.model, 'predict'):
            try:
                # Sklearn-style interface (joblib)
                score = float(self.model.predict(X)[0])
            #else:
            except (TypeError, AttributeError):
                # XGBoost Booster interface
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
                score = float(self.model.predict(dmatrix)[0])
            
            # Clip to valid range [0, 1]
            score = np.clip(score, 0.0, 1.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predict scores for multiple windows"""
        return [self.predict(f) for f in features_list]
    
    def get_model_info(self) -> Dict:
        """Get model metadata"""
        return {
            'version': self.metadata.get('version', 'unknown'),
            'val_mae': self.metadata.get('final_val_mae', None),
            'val_r2': self.metadata.get('final_val_r2', None),
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'model_dir': str(self.model_dir)
        }