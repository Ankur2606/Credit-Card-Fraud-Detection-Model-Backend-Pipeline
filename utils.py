import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import os
from datetime import datetime
import pickle

class FraudDetectionSystem:
    def __init__(self, data_path=None):
        """Initialize the Fraud Detection System."""
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        print("FraudDetectionSystem initialized")
        
    def load_data(self, data_path=None):
        """Load the dataset from the specified path."""
        print(f"Attempting to load data from: {data_path if data_path else self.data_path}")
        if data_path:
            self.data_path = data_path
        
        if self.data_path:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            print(f"Column names: {list(self.df.columns)}")
            print(f"Data types:\n{self.df.dtypes}")
            print(f"First 5 rows:\n{self.df.head()}")
            return self.df
        else:
            print("ERROR: Data path not specified.")
            raise ValueError("Data path not specified.")
    
    def explore_data(self):
        """Perform basic exploratory data analysis."""
        print("Starting exploratory data analysis...")
        if self.df is None:
            print("ERROR: No data loaded. Call load_data() first.")
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Dataset Information:")
        print(self.df.info())
        
        print("\nMissing values check:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found.")
        
        fraud_counts = self.df["is_fraud"].value_counts()
        print("\nFraud Distribution:")
        print(fraud_counts)
        
        fraud_percentage = fraud_counts[1] / len(self.df) * 100
        print(f"\nFraud percentage: {fraud_percentage:.2f}%")
        
        print("\nSummary statistics for numerical columns:")
        print(self.df.describe())
        
        print("Creating fraud distribution plot...")
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=self.df["is_fraud"])
        plt.title("Fraud vs Non-Fraud Transactions")
        plt.xlabel("Is Fraud (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():,}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 10),
                       textcoords = 'offset points')
        
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        plot_path = 'plots/fraud_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Fraud distribution plot saved to {plot_path}")
        
        return fraud_counts
    
    def preprocess_data(self):
        """Preprocess the data for model training."""
        print("Starting data preprocessing...")
        if self.df is None:
            print("ERROR: No data loaded. Call load_data() first.")
            raise ValueError("No data loaded. Call load_data() first.")
        
        df_processed = self.df.copy()
        print(f"Original dataframe shape: {df_processed.shape}")
        
        # Drop unnecessary columns
        columns_to_drop = ["Unnamed: 0", "first", "last", "street", "city", "state", 
                          "zip", "dob", "trans_num"]
        columns_actually_dropped = [col for col in columns_to_drop if col in df_processed.columns]
        print(f"Dropping columns: {columns_actually_dropped}")
        df_processed = df_processed.drop(columns=columns_actually_dropped)
        print(f"Dataframe shape after dropping columns: {df_processed.shape}")
        
        # Process datetime features
        print("Processing datetime features...")
        print(f"Sample trans_date_trans_time values: {df_processed['trans_date_trans_time'].head()}")
        df_processed["trans_date_trans_time"] = pd.to_datetime(df_processed["trans_date_trans_time"])
        df_processed["hour"] = df_processed["trans_date_trans_time"].dt.hour
        df_processed["day"] = df_processed["trans_date_trans_time"].dt.day
        df_processed["month"] = df_processed["trans_date_trans_time"].dt.month
        df_processed["dayofweek"] = df_processed["trans_date_trans_time"].dt.dayofweek
        print("Extracted datetime features sample:")
        print(df_processed[["hour", "day", "month", "dayofweek"]].head())
        df_processed = df_processed.drop(columns=["trans_date_trans_time", "unix_time"])
        
        # Calculate distance between customer and merchant
        print("Calculating geographic distance...")
        print("Sample coordinates values:")
        print(df_processed[["lat", "long", "merch_lat", "merch_long"]].head())
        df_processed['distance'] = np.sqrt(
            (df_processed['lat'] - df_processed['merch_lat'])**2 + 
            (df_processed['long'] - df_processed['merch_long'])**2
        )
        print(f"Distance calculation sample: {df_processed['distance'].head()}")
        
        # Encode categorical columns
        categorical_cols = ["merchant", "category", "job", "gender"]
        print(f"Encoding categorical columns: {categorical_cols}")
        for col in categorical_cols:
            if col in df_processed.columns:
                print(f"Encoding {col} with {df_processed[col].nunique()} unique values")
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                print(f"Sample encoded {col} values: {df_processed[col].head()}")
        
        # Feature scaling
        numeric_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "distance"]
        print(f"Scaling numeric columns: {numeric_cols}")
        print("Before scaling sample:")
        print(df_processed[numeric_cols].describe().loc[['mean', 'std']])
        self.scaler = StandardScaler()
        df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        print("After scaling sample:")
        print(df_processed[numeric_cols].describe().loc[['mean', 'std']])
        
        # Separate features and target
        print("Separating features and target...")
        X = df_processed.drop(columns=["is_fraud"])
        y = df_processed["is_fraud"]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Feature columns: {list(X.columns)}")
        print(f"Target distribution: {y.value_counts()}")
        
        # Split data into training and testing sets
        print("Splitting data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Testing features shape: {self.X_test.shape}")
        print(f"Training target distribution: {pd.Series(self.y_train).value_counts()}")
        print(f"Testing target distribution: {pd.Series(self.y_test).value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_imbalance(self, sampling_strategy=0.5):
        """Handle class imbalance using SMOTE."""
        print(f"Handling class imbalance with SMOTE (sampling_strategy={sampling_strategy})...")
        if self.X_train is None or self.y_train is None:
            print("ERROR: Data not split. Call preprocess_data() first.")
            raise ValueError("Data not split. Call preprocess_data() first.")
        
        print("Original class distribution:")
        original_distribution = pd.Series(self.y_train).value_counts()
        print(original_distribution)
        
        # Apply SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
        
        resampled_distribution = pd.Series(self.y_train_resampled).value_counts()
        print("Resampled class distribution:")
        print(resampled_distribution)
        print(f"Original shape: {self.X_train.shape}, Resampled shape: {self.X_train_resampled.shape}")
        
        return self.X_train_resampled, self.y_train_resampled
    
    def train_models(self):
        """Train multiple machine learning models."""
        print("Starting model training...")
        if not hasattr(self, 'X_train_resampled') or not hasattr(self, 'y_train_resampled'):
            print("ERROR: Resampled data not created. Call handle_imbalance() first.")
            raise ValueError("Resampled data not created. Call handle_imbalance() first.")
        
        # Random Forest
        print("Training Random Forest...")
        print("Random Forest parameters: n_estimators=100, random_state=42, n_jobs=-1")
        rf_start_time = datetime.now()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train_resampled, self.y_train_resampled)
        rf_end_time = datetime.now()
        print(f"Random Forest training complete. Training time: {rf_end_time - rf_start_time}")
        self.models['random_forest'] = rf
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        print("Gradient Boosting parameters: n_estimators=100, random_state=42")
        gb_start_time = datetime.now()
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train_resampled, self.y_train_resampled)
        gb_end_time = datetime.now()
        print(f"Gradient Boosting training complete. Training time: {gb_end_time - gb_start_time}")
        self.models['gradient_boosting'] = gb
        
        print(f"All models trained successfully. Available models: {list(self.models.keys())}")
        return self.models
    
    def evaluate_models(self):
        """Evaluate the trained models on the test set."""
        print("Starting model evaluation...")
        if not self.models:
            print("ERROR: No models trained. Call train_models() first.")
            raise ValueError("No models trained. Call train_models() first.")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}...")
            
            # Make predictions
            print(f"Making predictions with {name}...")
            start_time = datetime.now()
            y_pred = model.predict(self.X_test)
            end_time = datetime.now()
            print(f"Prediction time: {end_time - start_time}")
            
            # Calculate metrics
            print("Calculating performance metrics...")
            f1 = f1_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            
            print(f"\nConfusion Matrix for {name}:")
            print(cm)
            print("\nConfusion Matrix Interpretation:")
            print(f"True Negatives: {cm[0][0]}")
            print(f"False Positives: {cm[0][1]}")
            print(f"False Negatives: {cm[1][0]}")
            print(f"True Positives: {cm[1][1]}")
            
            print(f"\n{name} F1-Score: {f1:.4f}")
            print(f"{name} Classification Report:\n{report}")
            
            # Feature importance for applicable models
            if hasattr(model, 'feature_importances_'):
                print(f"Calculating feature importance for {name}...")
                feature_imp = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print(f"Top 10 features for {name}:")
                print(feature_imp.head(10))
                
                # Plot feature importance
                print(f"Creating feature importance plot for {name}...")
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
                plt.title(f'Top 15 Feature Importance - {name.replace("_", " ").title()}')
                plt.tight_layout()
                
                os.makedirs('plots', exist_ok=True)
                plot_path = f'plots/feature_importance_{name}.png'
                plt.savefig(plot_path)
                plt.close()
                print(f"Feature importance plot saved to {plot_path}")
            else:
                print(f"Model {name} does not support feature importance.")
                feature_imp = None
            
            results[name] = {
                'f1_score': f1,
                'classification_report': report,
                'confusion_matrix': cm,
                'feature_importance': feature_imp
            }
        
        # Print overall comparison
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY:")
        print("="*50)
        for name, result in results.items():
            print(f"{name}: F1-Score = {result['f1_score']:.4f}")
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save trained models and preprocessing objects."""
        print(f"Saving models to directory: {output_dir}")
        if not self.models:
            print("ERROR: No models trained. Call train_models() first.")
            raise ValueError("No models trained. Call train_models() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Using timestamp for versioning: {timestamp}")
        
        saved_paths = {}
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{name}_{timestamp}.pkl")
            print(f"Saving {name} model to {model_path}...")
            joblib.dump(model, model_path)
            saved_paths[name] = model_path
            print(f"Model '{name}' saved successfully")
        
        # Save preprocessing objects
        print("Saving preprocessing objects...")
        preproc = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': list(self.X_train.columns)
        }
        preproc_path = os.path.join(output_dir, f"preprocessing_{timestamp}.pkl")
        with open(preproc_path, 'wb') as f:
            pickle.dump(preproc, f)
        print(f"Preprocessing objects saved to {preproc_path}")
        
        result = {
            'models': saved_paths,
            'preprocessing': preproc_path
        }
        
        print("All models and preprocessing objects saved successfully")
        return result
    
    def load_models(self, model_paths, preproc_path):
        """Load trained models and preprocessing objects."""
        print(f"Loading models from: {model_paths}")
        print(f"Loading preprocessing objects from: {preproc_path}")
        
        self.models = {}
        
        # Load models
        for name, path in model_paths.items():
            print(f"Loading {name} model from {path}...")
            try:
                self.models[name] = joblib.load(path)
                print(f"Model '{name}' loaded successfully")
            except Exception as e:
                print(f"ERROR loading model '{name}': {str(e)}")
                raise
        
        # Load preprocessing objects
        print("Loading preprocessing objects...")
        try:
            with open(preproc_path, 'rb') as f:
                preproc = pickle.load(f)
            
            self.scaler = preproc['scaler']
            self.label_encoders = preproc['label_encoders']
            self.feature_columns = preproc['feature_columns']
            
            print("Preprocessing objects loaded successfully")
            print(f"Loaded {len(self.label_encoders)} label encoders")
            print(f"Feature columns: {self.feature_columns[:5]}... (total: {len(self.feature_columns)})")
        except Exception as e:
            print(f"ERROR loading preprocessing objects: {str(e)}")
            raise
        
        return self.models
    
    def predict(self, data, model_name='random_forest'):
        """
        Make fraud predictions on new data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            New transaction data to predict on
        model_name : str
            Name of model to use for prediction
            
        Returns:
        --------
        predictions : np.array
            Binary predictions (0 for legitimate, 1 for fraudulent)
        """
        print(f"Making predictions using {model_name} model...")
        print(f"Input data shape: {data.shape}")
        print(f"Input data columns: {list(data.columns)}")
        
        if model_name not in self.models:
            error_msg = f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Preprocess the data
        print("Preprocessing input data...")
        processed_data = self._preprocess_new_data(data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Make predictions
        print(f"Running {model_name} prediction...")
        model = self.models[model_name]
        predictions = model.predict(processed_data)
        
        # Summarize predictions
        pred_counts = pd.Series(predictions).value_counts()
        print("Prediction results:")
        print(pred_counts)
        fraud_percentage = (pred_counts.get(1, 0) / len(predictions) * 100) if len(predictions) > 0 else 0
        print(f"Predicted fraud percentage: {fraud_percentage:.2f}%")
        
        return predictions
    
    def predict_proba(self, data, model_name='random_forest'):
        """
        Get probability estimates for fraud predictions.
        
        Parameters:
        -----------
        data : pd.DataFrame
            New transaction data to predict on
        model_name : str
            Name of model to use for prediction
            
        Returns:
        --------
        probabilities : np.array
            Probability estimates for each class (legitimate and fraudulent)
        """
        print(f"Making probability predictions using {model_name} model...")
        print(f"Input data shape: {data.shape}")
        
        if model_name not in self.models:
            error_msg = f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Preprocess the data
        print("Preprocessing input data...")
        processed_data = self._preprocess_new_data(data)
        
        # Make probability predictions
        print(f"Running {model_name} probability prediction...")
        model = self.models[model_name]
        probabilities = model.predict_proba(processed_data)
        
        # Summarize probability distribution
        fraud_probs = probabilities[:, 1]
        print(f"Fraud probability statistics:")
        print(f"Min: {fraud_probs.min():.4f}, Max: {fraud_probs.max():.4f}, Mean: {fraud_probs.mean():.4f}")
        print(f"Number of transactions with fraud probability > 0.5: {(fraud_probs > 0.5).sum()}")
        print(f"Number of transactions with fraud probability > 0.9: {(fraud_probs > 0.9).sum()}")
        
        return probabilities
    
    def _preprocess_new_data(self, data):
        """Preprocess new data for prediction using saved preprocessing objects."""
        print("Starting preprocessing of new data...")
        if not self.scaler or not self.label_encoders:
            error_msg = "Preprocessing objects not loaded. Use load_models() first."
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        df = data.copy()
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Process datetime features if present
        if 'trans_date_trans_time' in df.columns:
            print("Processing datetime features...")
            print(f"Sample trans_date_trans_time values: {df['trans_date_trans_time'].head()}")
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
            df["hour"] = df["trans_date_trans_time"].dt.hour
            df["day"] = df["trans_date_trans_time"].dt.day
            df["month"] = df["trans_date_trans_time"].dt.month
            df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
            df = df.drop(columns=["trans_date_trans_time"])
            print("Datetime features extracted")
        
        # Drop unix_time if present
        if 'unix_time' in df.columns:
            print("Dropping unix_time column...")
            df = df.drop(columns=["unix_time"])
        
        # Calculate distance between customer and merchant if coordinates are present
        distance_cols = ['lat', 'long', 'merch_lat', 'merch_long']
        if all(col in df.columns for col in distance_cols):
            print("Calculating geographic distance...")
            df['distance'] = np.sqrt(
                (df['lat'] - df['merch_lat'])**2 + 
                (df['long'] - df['merch_long'])**2
            )
            print(f"Distance calculation sample: {df['distance'].head()}")
        else:
            missing = [col for col in distance_cols if col not in df.columns]
            print(f"Cannot calculate distance, missing columns: {missing}")
        
        # Apply label encoding to categorical columns
        print("Applying label encoding to categorical columns...")
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                print(f"Encoding column: {col}")
                df[col] = df[col].apply(lambda x: str(x))
                
                unseen = set(df[col].unique()) - set(encoder.classes_)
                if unseen:
                    print(f"Found {len(unseen)} unseen categories in {col}: {list(unseen)[:5]}...")
                
                df[col] = df[col].apply(lambda x: 'unknown' if x not in encoder.classes_ else x)
                
                known_mask = df[col].isin(encoder.classes_)
                print(f"Known values in {col}: {known_mask.sum()}/{len(df)}")
                
                df.loc[known_mask, col] = encoder.transform(df.loc[known_mask, col])
                
                df.loc[~known_mask, col] = -1
                print(f"Column {col} encoded. Sample values: {df[col].head()}")
        
        # Apply scaling to numeric columns
        print("Applying scaling to numeric columns...")
        numeric_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "distance"]
        for col in numeric_cols:
            if col in df.columns:
                print(f"Scaling column: {col}")
                before_scaling = df[col].describe()
                df[col] = self.scaler.transform(df[[col]])
                after_scaling = df[col].describe()
                print(f"Before scaling - mean: {before_scaling['mean']:.4f}, std: {before_scaling['std']:.4f}")
                print(f"After scaling - mean: {after_scaling['mean']:.4f}, std: {after_scaling['std']:.4f}")
        
        # Make sure all expected columns are present in the correct order
        required_cols = set(self.feature_columns)
        missing_cols = required_cols - set(df.columns)
        extra_cols = set(df.columns) - required_cols
        
        if missing_cols:
            print(f"Adding {len(missing_cols)} missing columns: {list(missing_cols)}")
            for col in missing_cols:
                df[col] = 0
        
        if extra_cols:
            print(f"Found {len(extra_cols)} extra columns that will be ignored: {list(extra_cols)}")
        
        # Select only the columns needed by the model in the correct order
        print(f"Reordering columns to match training data (columns count: {len(self.feature_columns)})")
        df = df[self.feature_columns]
        print(f"Final processed data shape: {df.shape}")
        
        return df
