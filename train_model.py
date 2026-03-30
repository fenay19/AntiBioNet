import os
import joblib
from controllers.data_controller import DataController
from controllers.model_controller import ModelController
from controllers.report_controller import ReportController

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Bacteria_dataset_Multiresictance.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.pkl")

def run_training_pipeline():
    print("--- Starting Offline Pipeline ---")
    data_ctrl = DataController(filepath=DATA_PATH)
    data_ctrl.load()
    data_ctrl.preprocess()
    data_ctrl.engineer_features()
    dataset = data_ctrl.get_dataset()

    model_ctrl = ModelController(dataset)
    model_ctrl.train_all()
    model_ctrl.evaluate_all()
    results = model_ctrl.get_results()

    report_ctrl = ReportController(dataset, results)
    
    # Save the necessary artifacts to pipeline.pkl
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    pipeline_data = {
        "dataset": dataset,
        "results": results,
        "report_ctrl": report_ctrl
    }
    
    print(f"\n[train_model.py] Saving pipeline artifacts to '{PIPELINE_PATH}'...")
    joblib.dump(pipeline_data, PIPELINE_PATH)
    print("Done! Application is ready to run.")
    
if __name__ == "__main__":
    run_training_pipeline()
