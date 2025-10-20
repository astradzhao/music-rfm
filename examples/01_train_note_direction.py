"""
Example: Train a control direction for musical notes

This example demonstrates how to train an RFM direction to control
a specific musical note (e.g., note C vs other notes) in music generation, using the Syntheory dataset.
"""

import random
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from datasets import load_dataset
from musicrfm import MusicGenController
import os
from musicrfm.utils import make_json_serializable
import json

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_NOTE = "D#"  # The note we want to control
    NUM_EXAMPLES = -1  # Number of examples per class (positive/negative), -1 for all
    N_COMPONENTS = 12  # Number of RFM components
    RFM_ITERS = 30  # Number of RFM iterations
    
    print(f"Training control direction for note: {TARGET_NOTE}")
    print(f"Device: {DEVICE}")
    
    # Load models
    print("\nLoading models...")
    music_model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large"
    ).to(DEVICE)
    music_processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(DEVICE)
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
    
    # Create controller
    print("\nCreating MusicGenController...")
    controller = MusicGenController(
        music_model,
        music_processor,
        encodec_model,
        encodec_processor,
        control_method="music_rfm",
        n_components=N_COMPONENTS,
        rfm_iters=RFM_ITERS,
        batch_size=16
    )
    
    # Load dataset
    print(f"\nLoading Syntheory dataset...")
    dataset = load_dataset("meganwei/syntheory", "notes")["train"] # options: notes, chords, scales, time_signatures, intervals, simple_progressions, tempos
    
    # Split data into positive (note C) and negative (other notes)
    print(f"\nPreparing data for note: {TARGET_NOTE}")
    positive_examples = [x for x in dataset if x["root_note_name"] == TARGET_NOTE]
    negative_examples = [x for x in dataset if x["root_note_name"] != TARGET_NOTE]
    
    print(f"Found {len(positive_examples)} positive examples")
    print(f"Found {len(negative_examples)} negative examples")
    
    # Sample and split into train/val/test (70/15/15)
    num_examples = min(len(positive_examples), len(negative_examples), NUM_EXAMPLES) if NUM_EXAMPLES != -1 else min(len(positive_examples), len(negative_examples))
    positive_samples = random.sample(positive_examples, num_examples)
    negative_samples = random.sample(negative_examples, num_examples)
    
    # Split into train (70%), val (15%), and test (15%)
    n_train_pos = int(0.7 * num_examples)
    n_val_pos = int(0.15 * num_examples)
    n_train_neg = int(0.7 * num_examples)
    n_val_neg = int(0.15 * num_examples)
    
    train_samples = positive_samples[:n_train_pos] + negative_samples[:n_train_neg]
    val_samples = positive_samples[n_train_pos:n_train_pos+n_val_pos] + negative_samples[n_train_neg:n_train_neg+n_val_neg]
    test_samples = positive_samples[n_train_pos+n_val_pos:] + negative_samples[n_train_neg+n_val_neg:]
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Extract audio features
    print("\nExtracting audio features...")
    train_features = [controller.get_audio_features(x) for x in train_samples]
    val_features = [controller.get_audio_features(x) for x in val_samples]
    test_features = [controller.get_audio_features(x) for x in test_samples]
    
    # Create labels (1 for target note, 0 for others)
    train_labels = torch.tensor(
        [1 if x["root_note_name"] == TARGET_NOTE else 0 for x in train_samples]
    ).reshape(-1, 1)
    val_labels = torch.tensor(
        [1 if x["root_note_name"] == TARGET_NOTE else 0 for x in val_samples]
    ).reshape(-1, 1)
    test_labels = torch.tensor(
        [1 if x["root_note_name"] == TARGET_NOTE else 0 for x in test_samples]
    ).reshape(-1, 1)
    
    # Concatenate features
    train_data = torch.cat(train_features, dim=0)
    val_data = torch.cat(val_features, dim=0)
    test_data = torch.cat(test_features, dim=0)
    
    # Train directions
    print("\nComputing control directions...")
    test_predictor_accs, test_direction_accs, results = controller.compute_directions(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        hidden_layers=list(range(-1, -48, -1)),  # All decoder layers
        tuning_metric='auc',
        pooling='mean',
        hyperparam_samples=100
    )
    
    results['n_components'] = N_COMPONENTS
    results['test_predictor_accs'] = test_predictor_accs
    results['test_direction_accs'] = test_direction_accs
    
    # Print results
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"Test Predictor Accuracy: {test_predictor_accs}")
    print(f"Test Direction Accuracy: {test_direction_accs}")
    
    # Save the directions
    concept_name = f"note_{TARGET_NOTE}_ncomp{N_COMPONENTS}"
    output_dir = "./trained_concepts"
    direction_path = os.path.join(output_dir, "directions")
    results_path = os.path.join(output_dir, "results")
    generations_path = os.path.join(output_dir, "generations")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(direction_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(generations_path, exist_ok=True)
    
    # Save results
    results_file = os.path.join(results_path, f"{concept_name}.json")
    with open(results_file, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=4)

    print(f"\nSaving directions to: {direction_path}/{concept_name}")
    controller.save(
        concept=concept_name,
        model_name="musicgen_large",
        path=direction_path
    )
    
    print("\n✓ Training complete!")
    print(f"  Saved concept: {concept_name}")
    print(f"  Location: {output_dir}/{concept_name}")
    print("\nYou can now use this concept for controlled generation.")
    print("See 02_generate_controlled.py for an example.")

if __name__ == "__main__":
    main()

