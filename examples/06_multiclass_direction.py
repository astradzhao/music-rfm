"""
Example: Train a multiclass direction for musical notes

This example demonstrates how to train an RFM direction for multiclass
classification of all musical notes (C, C#, D, etc.) using the Syntheory dataset.
"""

import random
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from datasets import load_dataset
from musicrfm import MusicGenController
from musicrfm.utils import make_json_serializable
import os
import json

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_COMPONENTS = 12  # Number of RFM components
    RFM_ITERS = 30  # Number of RFM iterations
    NUM_EXAMPLES = -1  # Number of examples per class (-1 for all available)
    TRAIN_SPLIT = 0.70  # Fraction of data for training
    VAL_SPLIT = 0.15  # Fraction of data for validation
    TEST_SPLIT = 0.15  # Fraction of data for testing
    
    print(f"Training multiclass direction for all musical notes")
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
    print(f"\nLoading Syntheory notes dataset...")
    dataset = load_dataset("meganwei/syntheory", "notes")["train"]
    
    # Get unique note values
    column = "root_note_name"
    unique_values = sorted(list(set(x[column] for x in dataset)))
    value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
    num_classes = len(unique_values)
    
    print(f"\nFound {num_classes} unique notes: {unique_values}")
    print(f"Note to index mapping: {value_to_idx}")
    
    # Collect examples for each class
    class_examples = {value: [] for value in unique_values}
    for example in dataset:
        value = example[column]
        class_examples[value].append(example)
    
    # Sample equal number of examples from each class
    min_examples = min(len(examples) for examples in class_examples.values())
    num_examples = min(NUM_EXAMPLES, min_examples) if NUM_EXAMPLES != -1 else min_examples
    
    print(f"\nUsing {num_examples} examples per class")
    for value, examples in class_examples.items():
        print(f"  {value}: {len(examples)} available, using {num_examples}")
    
    # Sample from each class
    sampled_examples = []
    for value in unique_values:
        class_samples = random.sample(class_examples[value], num_examples)
        sampled_examples.extend(class_samples)
    
    # Shuffle all examples
    random.shuffle(sampled_examples)
    
    # Split into train/val/test (70/15/15)
    n_total = len(sampled_examples)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    train_samples = sampled_examples[:n_train]
    val_samples = sampled_examples[n_train:n_train+n_val]
    test_samples = sampled_examples[n_train+n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    # Extract audio features in batches
    print("\nExtracting audio features...")
    batch_size = 32
    
    train_features = []
    for i in range(0, len(train_samples), batch_size):
        batch = train_samples[i:i+batch_size]
        if i % (batch_size * 5) == 0:
            print(f"  Processing train batch {i//batch_size + 1}/{(len(train_samples) + batch_size - 1)//batch_size}")
        batch_features = [controller.get_audio_features(x) for x in batch]
        train_features.extend(batch_features)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    val_features = []
    for i in range(0, len(val_samples), batch_size):
        batch = val_samples[i:i+batch_size]
        if i % (batch_size * 5) == 0:
            print(f"  Processing val batch {i//batch_size + 1}/{(len(val_samples) + batch_size - 1)//batch_size}")
        batch_features = [controller.get_audio_features(x) for x in batch]
        val_features.extend(batch_features)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    test_features = []
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        if i % (batch_size * 5) == 0:
            print(f"  Processing test batch {i//batch_size + 1}/{(len(test_samples) + batch_size - 1)//batch_size}")
        batch_features = [controller.get_audio_features(x) for x in batch]
        test_features.extend(batch_features)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create one-hot encoded labels
    print("\nCreating one-hot encoded labels...")
    train_labels = []
    for x in train_samples:
        value = x[column]
        idx = value_to_idx[value]
        one_hot = [0] * num_classes
        one_hot[idx] = 1
        train_labels.append(one_hot)
    
    val_labels = []
    for x in val_samples:
        value = x[column]
        idx = value_to_idx[value]
        one_hot = [0] * num_classes
        one_hot[idx] = 1
        val_labels.append(one_hot)
    
    test_labels = []
    for x in test_samples:
        value = x[column]
        idx = value_to_idx[value]
        one_hot = [0] * num_classes
        one_hot[idx] = 1
        test_labels.append(one_hot)
    
    # Convert to tensors
    train_data = torch.cat(train_features, dim=0)
    val_data = torch.cat(val_features, dim=0)
    test_data = torch.cat(test_features, dim=0)
    
    del train_features, val_features, test_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    
    print(f"\nFinal data shapes:")
    print(f"  Train data: {train_data.shape}, Train labels: {train_labels.shape}")
    print(f"  Val data: {val_data.shape}, Val labels: {val_labels.shape}")
    print(f"  Test data: {test_data.shape}, Test labels: {test_labels.shape}")
    
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
        classification=True,
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
    print(f"Results: {results}")
    
    # Save the directions
    concept_name = f"notes_multiclass_ncomp{N_COMPONENTS}"
    output_dir = "./trained_concepts/multiclass_directions"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving directions to: {output_dir}/{concept_name}")
    controller.save(
        concept=concept_name,
        model_name="musicgen_large",
        path=output_dir
    )
    
    results_file = f"{output_dir}/{concept_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=4)
    print(f"  Saved results: {results_file}")

if __name__ == "__main__":
    main()

