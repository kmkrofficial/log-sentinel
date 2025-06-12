import argparse
from config import DEFAULT_HYPERPARAMETERS, DB_PATH
from utils.database_manager import DatabaseManager
from engine.training_controller import TrainingController

def main():
    """
    Main entry point for starting a training run from the command line.
    """
    parser = argparse.ArgumentParser(description="Run LogSentinel Training")
    parser.add_argument(
        "--model", 
        type=str, 
        default="princeton-nlp/Sheared-Llama-1.3B",
        help="Name of the model to use (local directory or Hugging Face ID)."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="BGL",
        help="Name of the dataset directory inside the 'datasets' folder."
    )
    args = parser.parse_args()

    # --- Configuration for the Run ---
    model_name = args.model
    dataset_name = args.dataset
    hyperparameters = DEFAULT_HYPERPARAMETERS
    
    # --- Initialize Managers and Controller ---
    db_manager = DatabaseManager(db_path=DB_PATH)
    
    print("--- LogSentinel Training Run ---")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print("---------------------------------")

    controller = TrainingController(
        model_name=model_name,
        dataset_name=dataset_name,
        hyperparameters=hyperparameters,
        db_manager=db_manager
    )

    # --- Execute the Run ---
    try:
        run_id = controller.run()
        if run_id:
            print(f"\nSuccessfully completed run with ID: {run_id}")
        else:
            print("\nRun failed to start or was a duplicate.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: The training run failed. Reason: {e}")
    finally:
        db_manager.close()

if __name__ == '__main__':
    main()