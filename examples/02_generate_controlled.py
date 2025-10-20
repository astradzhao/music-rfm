"""
Example: Generate music with a trained control direction

This example shows how to load a pre-trained control direction
and use it to generate music with controlled attributes.

Includes advanced options:
- Layer selection (top-k, threshold, exponential dropout)
- Time-varying control
- Probabilistic injection
- Custom control coefficients
"""

import torch
import soundfile as sf
import os
import json
import math
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController

def select_top_k_layers(results, k, regression=False):
    """
    Select the top k layers by performance.
    
    Args:
        results: Results dictionary containing layer_metrics
        k: Number of top layers to select
        regression: Whether this is a regression task
        
    Returns:
        List of layer numbers sorted by layer index
    """
    if regression:
        # For regression, lower MSE/loss is better
        sorted_layers = sorted(results['layer_metrics'], 
                             key=lambda x: results['layer_metrics'][x]['train_results']['score'], 
                             reverse=False)
    else:
        # For classification, higher accuracy/AUC is better
        sorted_layers = sorted(results['layer_metrics'], 
                             key=lambda x: results['layer_metrics'][x]['train_results']['score'], 
                             reverse=True)
    
    layers_to_control = [int(x) for x in sorted_layers[:k]]
    layers_to_control.sort()
    
    print(f"Selected top {k} layers: {layers_to_control}")
    return layers_to_control

def select_exponential_layer_dropout(results, regression=False, base_weight=1.0, decay_rate=0.95):
    """
    Select all layers with exponentially decreasing weights based on performance.
    Better layers get higher weights.
    
    Args:
        results: Results dictionary containing layer_metrics
        regression: Whether this is a regression task
        base_weight: Base weight for the best layer
        decay_rate: Rate at which weights decay
        
    Returns:
        Tuple of (layers_to_control, layer_weights)
    """
    layer_scores = {}
    for layer_str, metrics in results['layer_metrics'].items():
        layer_num = int(layer_str)
        score = metrics['train_results']['score']
        layer_scores[layer_num] = score
    
    # Normalize scores to [0, 1]
    if regression:
        best_score = min(layer_scores.values())
        worst_score = max(layer_scores.values())
        normalized_scores = {layer: (worst_score - score) / (worst_score - best_score) 
                           for layer, score in layer_scores.items()}
    else:
        best_score = max(layer_scores.values())
        worst_score = min(layer_scores.values())
        normalized_scores = {layer: (score - worst_score) / (best_score - worst_score) 
                           for layer, score in layer_scores.items()}
    
    # Calculate exponential weights
    layers_to_control = []
    layer_weights = []
    
    for layer_num in sorted(layer_scores.keys()):
        normalized_score = normalized_scores[layer_num]
        weight = base_weight * (normalized_score ** (1/decay_rate))
        layers_to_control.append(layer_num)
        layer_weights.append(weight)
    
    print(f"Using exponential layer dropout with {len(layers_to_control)} layers")
    print(f"Top 5 layers by weight:")
    sorted_by_weight = sorted(zip(layers_to_control, layer_weights), key=lambda x: x[1], reverse=True)
    for i in range(min(5, len(sorted_by_weight))):
        layer, weight = sorted_by_weight[i]
        print(f"  Layer {layer}: weight={weight:.4f}")
    
    return layers_to_control, layer_weights

def exponential_decay(t, base_coef, decay_rate=0.998):
    """
    Exponential decay function for time-varying control.
    
    Args:
        t: Current time step
        base_coef: Base control coefficient
        decay_rate: Decay rate (closer to 1 = slower decay)
        
    Returns:
        Scaled control coefficient
    """
    return base_coef * (decay_rate ** t)

def linear_decay(t, base_coef, total_steps=1500):
    """
    Linear decay function for time-varying control.
    
    Args:
        t: Current time step
        base_coef: Base control coefficient
        total_steps: Total number of steps
        
    Returns:
        Scaled control coefficient
    """
    return base_coef * (1 - min(max(t / total_steps, 0), 1))

def main():
    # ==================== Configuration ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONCEPT_NAME = "note_C_ncomp12"  # The concept to load
    DIRECTIONS_PATH = "./trained_concepts/directions"
    RESULTS_PATH = "./trained_concepts/results"
    OUTPUT_DIR = "./trained_concepts/generations"
    BATCH_SIZE = 4  # Number of prompts to process at once
    
    # Test prompts
    test_prompts = [
        "A relaxing jazz song with piano",
        "Upbeat electronic dance music with synths",
        "Slow classical piano piece",
        "Fast-paced rock song with guitar"
    ]
    
    # ==================== Advanced Options ====================
    
    # Layer selection method (choose one):
    LAYER_SELECTION = "exp_weighting"  # Options: "all", "top_k", "exp_weighting"
    TOP_K = 16  # Number of top layers (if using "top_k")
    EXP_BASE_WEIGHT = 1.0  # Base weight for exponential dropout
    EXP_DECAY_RATE = 0.95  # Decay rate for exponential dropout
    
    # Control coefficients
    CONTROL_COEFFICIENTS = [0.3, 0.5, 0.7]
    
    # Time control (set to None for constant control)
    TIME_CONTROL = None  # Options: None, "exp_decay", "linear_decay"
    TIME_DECAY_RATE = 0.998  # For exponential decay
    
    # Probabilistic injection
    INJECT_CHANCE = 0.3  # Probability of applying control at each step (0.0-1.0)
    
    # Is this a regression concept?
    IS_REGRESSION = False  # Set to True for tempo/continuous concepts
    
    # ==================== Validate Paths and Load Results ====================
    
    print(f"Concept: {CONCEPT_NAME}")
    print(f"Layer selection: {LAYER_SELECTION}")
    print(f"Time control: {TIME_CONTROL}")
    print(f"Inject chance: {INJECT_CHANCE}")
    
    # Check if concept directory exists
    concept_dir_path = os.path.join(DIRECTIONS_PATH, f"music_rfm_{CONCEPT_NAME}_musicgen_large.pkl")
    if not os.path.exists(concept_dir_path):
        print(f"\n✗ Error: Concept not found at {DIRECTIONS_PATH}")
        print(f"   Looking for: music_rfm_{CONCEPT_NAME}_musicgen_large.pkl")
        print("\nPlease run 01_train_note_direction.py first to train a concept.")
        return
    
    # Load results file if needed for layer selection
    results = None
    results_file = os.path.join(RESULTS_PATH, f"{CONCEPT_NAME}.json")
    
    if LAYER_SELECTION in ["top_k", "exp_weighting"]:
        # Results file is required for these methods
        if not os.path.exists(results_file):
            print(f"\n✗ Error: Results file required for '{LAYER_SELECTION}' layer selection")
            print(f"   Looking for: {results_file}")
            print("\nOptions:")
            print("  1. Change LAYER_SELECTION to 'all' (doesn't require results file)")
            print("  2. Re-train the concept and ensure results are saved")
            return
        
        print(f"\nLoading results from: {results_file}")
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print("✓ Results loaded successfully")
        except json.JSONDecodeError as e:
            print(f"\n✗ Error: Invalid JSON in results file")
            print(f"   {str(e)}")
            return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==================== Load Models ====================
    print(f"\nDevice: {DEVICE}")
    print("Loading models (this may take a minute)...")
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
        n_components=12,  # Should match training
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
        layers_to_control = select_top_k_layers(results, TOP_K, IS_REGRESSION)
            
    elif LAYER_SELECTION == "exp_weighting":
        layers_to_control, layer_weights = select_exponential_layer_dropout(
            results, IS_REGRESSION, EXP_BASE_WEIGHT, EXP_DECAY_RATE
        )
    else:
        raise ValueError(f"Invalid layer selection method: {LAYER_SELECTION}")
    
    # ==================== Time Control Function ====================
    
    time_control_fn = None
    if TIME_CONTROL == "exp_decay":
        time_control_fn = lambda t, base: exponential_decay(t, base, TIME_DECAY_RATE)
        print(f"\nUsing exponential decay with rate {TIME_DECAY_RATE}")
    elif TIME_CONTROL == "linear_decay":
        time_control_fn = lambda t, base: linear_decay(t, base)
        print("\nUsing linear decay over 1500 steps")
    elif TIME_CONTROL is None:
        print("\nUsing constant control (no time variation)")
    
    # ==================== Create Output Directory ====================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sampling_rate = music_model.config.audio_encoder.sampling_rate
    
    print("\n" + "="*60)
    print("Generating Controlled Music")
    print("="*60)
    
    # ==================== Generate Music ====================
    
    # Process prompts in batches
    for i in range(0, len(test_prompts), BATCH_SIZE):
        batch_prompts = test_prompts[i:i+BATCH_SIZE]
        batch_indices = list(range(i, min(i+BATCH_SIZE, len(test_prompts))))
        
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}: prompts {i+1}-{min(i+BATCH_SIZE, len(test_prompts))}")
        for j, (idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
            print(f"  Prompt {idx+1}: '{prompt}'")
        
        # Generate baseline (no control) for entire batch
        print("  Generating baseline (no control)...")
        inputs = music_processor(
            text=batch_prompts,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            baseline_audios = music_model.generate(**inputs, max_new_tokens=1500)
        
        # Save each baseline
        for j, (prompt_idx, audio) in enumerate(zip(batch_indices, baseline_audios)):
            baseline_path = f"{OUTPUT_DIR}/prompt{prompt_idx}_baseline.flac"
            sf.write(baseline_path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
        print(f"    ✓ Saved {len(batch_indices)} baseline files")
        
        # Generate with different control strengths
        for control_coef in CONTROL_COEFFICIENTS:
            print(f"  Generating with control={control_coef}...")
            
            controlled_audios = controller.generate(
                batch_prompts,
                layers_to_control=layers_to_control,
                control_coef=control_coef,
                max_new_tokens=1500,
                time_control_fn=time_control_fn,
                layer_weights=layer_weights,
                inject_chance=INJECT_CHANCE
            )
            
            # Save each controlled output
            for j, (prompt_idx, audio) in enumerate(zip(batch_indices, controlled_audios)):
                # Build filename with config info
                filename_parts = [f"prompt{prompt_idx}", f"control{control_coef}"]
                if LAYER_SELECTION != "all":
                    filename_parts.append(LAYER_SELECTION)
                if TIME_CONTROL:
                    filename_parts.append(TIME_CONTROL)
                if INJECT_CHANCE < 1.0:
                    filename_parts.append(f"inject{INJECT_CHANCE}")
                
                output_path = f"{OUTPUT_DIR}/{'_'.join(filename_parts)}.flac"
                sf.write(output_path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
            print(f"    ✓ Saved {len(batch_indices)} controlled files")
    
    print("\n" + "="*60)
    print("✓ Generation complete!")
    print(f"  Generated {len(test_prompts) * (len(CONTROL_COEFFICIENTS) + 1)} audio files")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Print summary of configuration
    print("\nConfiguration used:")
    print(f"  Concept: {CONCEPT_NAME}")
    print(f"  Layers: {LAYER_SELECTION} ({len(layers_to_control)} layers)")
    print(f"  Time control: {TIME_CONTROL or 'None'}")
    print(f"  Inject chance: {INJECT_CHANCE}")
    print(f"  Control coefficients: {CONTROL_COEFFICIENTS}")
    print("\nCompare the baseline and controlled versions to hear the effect!")

if __name__ == "__main__":
    main()
