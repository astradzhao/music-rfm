"""
Example: Train and generate with a regression direction for tempo (BPM)

This example demonstrates how to:
1. Train an RFM direction to control tempo (beats per minute) 
2. Generate music with continuous tempo control

Uses the Syntheory dataset for training.
"""

import random
import torch
import json
import os
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from datasets import load_dataset
from musicrfm import MusicGenController
from musicrfm.utils import make_json_serializable

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def select_top_k_layers(results, k):
    """Select the top k layers by performance (lower MSE is better for regression)."""
    sorted_layers = sorted(results['layer_metrics'], 
                         key=lambda x: results['layer_metrics'][x]['train_results']['score'], 
                         reverse=False)  # Lower MSE is better
    
    layers_to_control = [int(x) for x in sorted_layers[:k]]
    layers_to_control.sort()
    
    print(f"Using top-k selection: top {k} layers")
    print(f"Layers to control: {layers_to_control}")
    print("Scores (MSE) of selected layers:")
    for layer in layers_to_control:
        score = results['layer_metrics'][str(layer)]['train_results']['score']
        print(f"Layer {layer}: {score:.4f}", end=", ")
    print()
    
    return layers_to_control

def select_exponential_layer_dropout(results, base_weight=1.0, decay_rate=0.95):
    """Select layers using exponential dropout based on performance (lower MSE is better)."""
    layer_scores = {}
    for layer_str, metrics in results['layer_metrics'].items():
        layer_num = int(layer_str)
        score = metrics['train_results']['score']
        layer_scores[layer_num] = score
    
    # For regression, lower MSE is better
    best_score = min(layer_scores.values())
    worst_score = max(layer_scores.values())
    
    # Invert scores so better (lower) scores get higher weights
    if worst_score == best_score:
        normalized_scores = {layer: 1.0 for layer in layer_scores.keys()}
    else:
        normalized_scores = {
            layer: (worst_score - score) / (worst_score - best_score) 
            for layer, score in layer_scores.items()
        }
    
    layers_to_control = []
    layer_weights = []
    
    for layer_num in sorted(layer_scores.keys()):
        normalized_score = normalized_scores[layer_num]
        weight = base_weight * (normalized_score ** (1/decay_rate))
        
        layers_to_control.append(layer_num)
        layer_weights.append(weight)
    
    print(f"Using exponential layer dropout: all {len(layers_to_control)} layers")
    print(f"Base weight: {base_weight}, Decay rate: {decay_rate}")
    print(f"Score range: {best_score:.4f} to {worst_score:.4f}")
    print("Top 10 layers and weights:")
    
    sorted_by_weight = sorted(
        zip(layers_to_control, layer_weights, [layer_scores[l] for l in layers_to_control]), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for i in range(min(10, len(sorted_by_weight))):
        layer, weight, score = sorted_by_weight[i]
        print(f"  Layer {layer}: weight={weight:.4f}, MSE={score:.4f}")
    if len(layers_to_control) > 10:
        print(f"  ... and {len(layers_to_control) - 10} more layers")
    
    return layers_to_control, layer_weights

def train_tempo_direction():
    """Train a regression direction for tempo control."""
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_COMPONENTS = 8  # Number of RFM components
    RFM_ITERS = 30  # Number of RFM iterations
    TRAIN_SPLIT = 0.70  # Fraction of data for training
    VAL_SPLIT = 0.15  # Fraction of data for validation
    TEST_SPLIT = 0.15  # Fraction of data for testing
    NORMALIZE = True  # Normalize tempo values
    
    print("=" * 60)
    print("Training Regression Direction for Tempo (BPM)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load models
    print("\nLoading models...")
    music_model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large"
    ).to(DEVICE)
    music_processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(DEVICE)
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
    print("✓ Models loaded successfully")
    
    # Create controller
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
    print(f"\nLoading Syntheory tempos dataset...")
    dataset = load_dataset("meganwei/syntheory", "tempos")["train"]
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=SEED)
    
    # Split into train/val/test (70/15/15)
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    
    train_samples = dataset.select(range(n_train))
    val_samples = dataset.select(range(n_train, n_train + n_val))
    test_samples = dataset.select(range(n_train + n_val, n_total))
    
    print(f"Dataset split:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    # Extract audio features
    print("\nExtracting audio features...")
    train_features = [controller.get_audio_features(x) for x in train_samples]
    val_features = [controller.get_audio_features(x) for x in val_samples]
    test_features = [controller.get_audio_features(x) for x in test_samples]
    
    # Get labels (BPM values)
    train_labels = torch.tensor(
        [x["bpm"] for x in train_samples],
        dtype=torch.float32
    ).reshape(-1, 1)
    val_labels = torch.tensor(
        [x["bpm"] for x in val_samples],
        dtype=torch.float32
    ).reshape(-1, 1)
    test_labels = torch.tensor(
        [x["bpm"] for x in test_samples],
        dtype=torch.float32
    ).reshape(-1, 1)
    
    # Normalize labels using training statistics
    if NORMALIZE:
        train_mean = train_labels.mean()
        train_std = train_labels.std()
        train_labels = (train_labels - train_mean) / train_std
        val_labels = (val_labels - train_mean) / train_std
        test_labels = (test_labels - train_mean) / train_std
        
        print(f"\nLabel normalization:")
        print(f"  Original - Mean: {train_mean:.3f}, Std: {train_std:.3f}")
        print(f"  Normalized - Train Mean: {train_labels.mean():.3f}, Train Std: {train_labels.std():.3f}")
    
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
        tuning_metric='mse',
        pooling='mean',
        regression=True,
        hyperparam_samples=10,
    )
    
    # Add metadata to results
    results['n_components'] = N_COMPONENTS
    results['test_predictor_accs'] = test_predictor_accs
    results['test_direction_accs'] = test_direction_accs
    
    # Setup output directories
    concept_name = f"tempo_regression_ncomp{N_COMPONENTS}_normalized" if NORMALIZE else f"tempo_regression_ncomp{N_COMPONENTS}"
    output_dir = "./trained_concepts"
    direction_path = os.path.join(output_dir, "directions")
    results_path = os.path.join(output_dir, "results")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(direction_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # Save directions
    print(f"\nSaving directions to: {direction_path}/{concept_name}")
    controller.save(
        concept=concept_name,
        model_name="musicgen_large",
        path=direction_path
    )
    
    # Save results JSON
    results_file = os.path.join(results_path, f"{concept_name}.json")
    
    # Include normalization stats in results
    if NORMALIZE:
        results['normalization_stats'] = {
            'mean': train_mean.item(),
            'std': train_std.item()
        }
    
    with open(results_file, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=4)
    
    print(f"  Saved results: {results_file}")
    
    print("\n✓ Training complete!")
    print(f"  Saved concept: {concept_name}")
    print(f"  Location: {direction_path}/{concept_name}")
    print("\nYou can now use generate_with_tempo() to generate tempo-controlled music.")

def generate_with_tempo():
    """Generate music with trained tempo regression direction."""
    # ==================== Configuration ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_COMPONENTS = 8  # Must match training
    CONCEPT_NAME = "tempo_regression_ncomp8_normalized"
    DIRECTIONS_PATH = "./trained_concepts/directions"
    RESULTS_PATH = "./trained_concepts/results"
    OUTPUT_DIR = "./trained_concepts/generations_tempo"
    BATCH_SIZE = 4
    
    test_prompts = [
        "A relaxing jazz song with piano",
        "Upbeat electronic dance music with synths",
        "Slow classical piano piece",
        "Fast-paced rock song with guitar"
    ]
    
    # ==================== Advanced Options ====================
    
    # Layer selection method
    LAYER_SELECTION = "exp_dropout"  # Options: "all", "top_k", "exp_dropout"
    TOP_K = 16
    EXP_BASE_WEIGHT = 1.0
    EXP_DECAY_RATE = 0.95
    
    # Control coefficients (for regression, negative = slower, positive = faster)
    CONTROL_COEFFICIENTS = [-0.4, -0.2, 0.0, 0.2, 0.4]
    
    # Time control
    INJECT_CHANCE = 0.3  # Probability of applying control at each step
    
    # ==================== Validation ====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = f"{RESULTS_PATH}/{CONCEPT_NAME}.json"

    results = None
    if LAYER_SELECTION in ["top_k", "exp_dropout"]:
        if not os.path.exists(results_file):
            print(f"✗ Error: Results file required for '{LAYER_SELECTION}' layer selection")
            print(f"   Looking for: {results_file}")
            print("\nPlease ensure the results file exists or use LAYER_SELECTION='all'")
            return
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✓ Loaded results from: {results_file}")
        except json.JSONDecodeError:
            print(f"✗ Error: Invalid JSON in results file: {results_file}")
            return
    
    # ==================== Load Models ====================
    print(f"\nDevice: {DEVICE}")
    print("Loading models...")
    music_model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large"
    ).to(DEVICE)
    music_processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(DEVICE)
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
    
    controller = MusicGenController(
        music_model,
        music_processor,
        encodec_model,
        encodec_processor,
        control_method="music_rfm",
        n_components=N_COMPONENTS,
        rfm_iters=30,
        batch_size=4
    )
    print("✓ Models loaded successfully")
    
    # Load the trained concept
    print(f"\nLoading trained concept: {CONCEPT_NAME}")
    controller.load(
        concept=CONCEPT_NAME,
        model_name="musicgen_large",
        path=DIRECTIONS_PATH
    )
    print("✓ Concept loaded successfully!")
    
    # ==================== Layer Selection ====================
    layer_weights = None
    
    if LAYER_SELECTION == "all":
        layers_to_control = list(range(-1, -48, -1))
        print(f"\nUsing all {len(layers_to_control)} layers")
        
    elif LAYER_SELECTION == "top_k":
        layers_to_control = select_top_k_layers(results, TOP_K)
        
    elif LAYER_SELECTION == "exp_dropout":
        layers_to_control, layer_weights = select_exponential_layer_dropout(
            results, EXP_BASE_WEIGHT, EXP_DECAY_RATE
        )
    
    # ==================== Generate Music ====================
    sampling_rate = music_model.config.audio_encoder.sampling_rate
    
    print("\n" + "="*60)
    print("Generating Music with Tempo Control")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Control coefficients: {CONTROL_COEFFICIENTS}")
    print(f"Inject chance: {INJECT_CHANCE}")
    
    # Generate in batches
    for i in range(0, len(test_prompts), BATCH_SIZE):
        batch_prompts = test_prompts[i:i+BATCH_SIZE]
        batch_indices = list(range(i, min(i+BATCH_SIZE, len(test_prompts))))
        
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}: prompts {i+1}-{min(i+BATCH_SIZE, len(test_prompts))}")
        for j, prompt in enumerate(batch_prompts):
            print(f"  Prompt {i+j+1}: '{prompt}'")
        
        # Generate baseline (no control)
        print("  Generating baseline (no control)...")
        inputs = music_processor(
            text=batch_prompts,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            baseline_audio = music_model.generate(**inputs, max_new_tokens=1500)
        
        # Save baseline
        for j, (prompt_idx, audio) in enumerate(zip(batch_indices, baseline_audio)):
            baseline_path = f"{OUTPUT_DIR}/prompt{prompt_idx}_baseline.flac"
            sf.write(baseline_path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
            print(f"    ✓ Saved baseline: {baseline_path}")
        
        # Generate with different control coefficients
        for control_coef in CONTROL_COEFFICIENTS:
            print(f"  Generating with control coefficient: {control_coef}")
            
            controlled_audio = controller.generate(
                batch_prompts,
                layers_to_control=layers_to_control,
                control_coef=control_coef,
                max_new_tokens=1500,
                layer_weights=layer_weights,
                inject_chance=INJECT_CHANCE
            )
            
            # Save each output in the batch
            for j, (prompt_idx, audio) in enumerate(zip(batch_indices, controlled_audio)):
                # Build filename
                filename_parts = [f"prompt{prompt_idx}"]
                filename_parts.append(f"control{control_coef}")
                
                if LAYER_SELECTION == "exp_dropout":
                    filename_parts.append(f"exp_dropout")
                elif LAYER_SELECTION == "top_k":
                    filename_parts.append(f"topk{TOP_K}")
                
                if INJECT_CHANCE < 1.0:
                    filename_parts.append(f"inject{INJECT_CHANCE}")
                
                output_path = f"{OUTPUT_DIR}/{'_'.join(filename_parts)}.flac"
                sf.write(output_path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
                print(f"    ✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("✓ Generation complete!")
    print(f"Audio files saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_tempo_direction()
        elif sys.argv[1] == "generate":
            generate_with_tempo()
        else:
            print("Usage:")
            print("  python 05_regression_direction.py train     # Train tempo direction")
            print("  python 05_regression_direction.py generate  # Generate with trained direction")
    else:
        # Default: run both
        print("Running both training and generation...")
        print("\n" + "="*80)
        print("STEP 1: Training Direction")
        print("="*80)
        train_tempo_direction()
        
        print("\n\n" + "="*80)
        print("STEP 2: Generating Music")
        print("="*80)
        generate_with_tempo()
