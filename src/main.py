import os
import pickle
from preprocessing.data_loader import get_data_generators
from models.final_model import build_final_model 
from training.train import train_model
from evaluation.evaluator import evaluate_model 
from utils.plot_results import plot_history

EMOTIONS = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6, 'contempt': 7
}

def main():
    print("\n" + "="*30)
    print("--- Phase_Two Started ---")
    print("="*30)

    csv_path = 'data/raw/dataset_cleaned.csv'
    img_dir = 'data/raw/'
    model_path = 'models/final_model.keras'
    history_path = 'models/phase2_history.pkl'
    summary_path = 'models/model_summary.txt'

    if not os.path.exists('models'):
        os.makedirs('models')

    print("\n--- Phase 1: Data Loading ---")
    train_gen, val_gen, test_gen = get_data_generators(csv_path, img_dir, batch_size=128)
    print(f"Target Emotions: {list(EMOTIONS.keys())}")

    print("\n--- Phase 2: Building Final Model ---")
    model = build_final_model(num_classes=len(EMOTIONS))
    
    model.summary() 
    
    print(f"Saving model summary to {summary_path}...")
    with open(summary_path, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print("\n--- Phase 3: Training ---")
    history = train_model(model, train_gen, val_gen, epochs=40)

    print(f"\nSaving training history to {history_path}...")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print("Plotting training history...")
    plot_history(history)

    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("Model saved.")

    print("\n--- Phase 4: Detailed Evaluation ---")
    evaluate_model(model, test_gen, EMOTIONS)
    
    print("\n" + "="*30)
    print("--- Phase_Two Ended ---")
    print("="*30)

if __name__ == "__main__":
    main()