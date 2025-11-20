from ml_pipeline.dataset.build_dataset import main as build
from ml_pipeline.training.train_model import main as train
from ml_pipeline.evaluation.evaluate_model import main as eval

if __name__ == "__main__":
    print("=== Building datasets ===")
    build()

    print("=== Training model ===")
    train()

    print("=== Evaluating model ===")
    eval()

    print("✔ COMPLETE PIPELINE FINISHED")
