from utils import *

# Main pipeline function
def run_fraud_detection_pipeline(data_path):
    print("\n" + "="*80)
    print("STARTING FRAUD DETECTION PIPELINE")
    print("="*80)
    print(f"Data path: {data_path}")
    
    # Step 1
    print("\nStep 1: Initializing Fraud Detection System...")
    fds = FraudDetectionSystem()
    
    # Step 2
    print("\nStep 2: Loading data...")
    fds.load_data(data_path)
    
    # Step 3
    print("\nStep 3: Exploring data...")
    fds.explore_data()
    
    # Step 4
    print("\nStep 4: Preprocessing data...")
    fds.preprocess_data()
    
    # Step 5
    print("\nStep 5: Handling class imbalance...")
    fds.handle_imbalance(sampling_strategy=0.5)
    
    # Step 6
    print("\nStep 6: Training models...")
    fds.train_models()
    
    # Step 7
    print("\nStep 7: Evaluating models...")
    results = fds.evaluate_models()
    
    # Step 8
    print("\nStep 8: Saving models...")
    model_paths = fds.save_models()
    
    print("\n" + "="*80)
    print("FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Summary
    print("\nPIPELINE SUMMARY:")
    print(f"- Data loaded: {fds.df.shape[0]} transactions")
    print(f"- Fraud cases: {fds.df['is_fraud'].sum()} ({fds.df['is_fraud'].mean() * 100:.2f}%)")
    print(f"- Models trained: {list(fds.models.keys())}")
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"- Best model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    print(f"- Models saved to directory: {os.path.abspath('models')}")
    
    return fds, results, model_paths

if __name__ == "__main__":
    print("FRAUD DETECTION SYSTEM EXECUTION STARTED")
    try:
        data_path = "fraudTrain.csv"
        print(f"Using dataset: {data_path}")
        fds, results, model_paths = run_fraud_detection_pipeline(data_path)
        print("FRAUD DETECTION SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"ERROR: FRAUD DETECTION SYSTEM EXECUTION FAILED: {str(e)}")
        import traceback
        print(traceback.format_exc())
