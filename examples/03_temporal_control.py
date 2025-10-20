"""
Example: Time-varying control during generation

This example demonstrates how to apply control that changes over time,
such as exponentially decaying the control strength or crossfading between
two different control directions.

Includes same advanced options as 02_generate_controlled.py:
- Layer selection (top-k, threshold, exponential dropout)
- Probabilistic injection
- Custom control coefficients
"""

import torch
import soundfile as sf
import os
import json
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController

def select_top_k_layers(results, k, regression=False):
    """Select the top k layers by performance."""
    if regression:
        sorted_layers = sorted(results['layer_metrics'], 
                             key=lambda x: results['layer_metrics'][x]['train_results']['score'], 
                             reverse=False)
    else:
        sorted_layers = sorted(results['layer_metrics'], 
                             key=lambda x: results['layer_metrics'][x]['train_results']['score'], 
                             reverse=True)
    
    layers_to_control = [int(x) for x in sorted_layers[:k]]
    layers_to_control.sort()
    print(f"Selected top {k} layers: {layers_to_control}")
    return layers_to_control

def select_exponential_layer_dropout(results, regression=False, base_weight=1.0, decay_rate=0.95):
    """Select all layers with exponentially decreasing weights based on performance."""
    layer_scores = {}
    for layer_str, metrics in results['layer_metrics'].items():
        layer_num = int(layer_str)
        score = metrics['train_results']['score']
        layer_scores[layer_num] = score
    
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

def exponential_decay(t, base, decay_rate=0.998):
    """Exponentially decay control strength over time"""
    return base * (decay_rate ** t)

def linear_ramp(t, base, start_time=0, end_time=1500):
    """Linearly increase control from 0 to base over time"""
    if t < start_time:
        return 0.0
    elif t >= end_time:
        return base
    else:
        progress = (t - start_time) / (end_time - start_time)
        return base * progress

def main():
    # ==================== Configuration ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONCEPT_NAME = "note_C_ncomp12"
    DIRECTIONS_PATH = "./trained_concepts/directions"
    RESULTS_PATH = "./trained_concepts/results"
    OUTPUT_DIR = "./trained_concepts/generations_temporal"
    BATCH_SIZE = 4
    
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
    
    # Control coefficient
    CONTROL_COEF = 0.6
    
    # Probabilistic injection
    INJECT_CHANCE = 1.0  # Probability of applying control at each step (0.0-1.0)
    
    # Is this a regression concept?
    IS_REGRESSION = False  # Set to True for tempo/continuous concepts
    
    # ==================== Validate Paths and Load Results ====================
    
    print(f"Temporal Control Example")
    print(f"Concept: {CONCEPT_NAME}")
    print(f"Layer selection: {LAYER_SELECTION}")
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
        n_components=12,
        rfm_iters=30,
        batch_size=BATCH_SIZE
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
    
    # ==================== Generate Music ====================
    
    sampling_rate = music_model.config.audio_encoder.sampling_rate
    
    print("\n" + "="*60)
    print("Generating with Temporal Control")
    print("="*60)
    
    # Process prompts in batches
    for i in range(0, len(test_prompts), BATCH_SIZE):
        batch_prompts = test_prompts[i:i+BATCH_SIZE]
        batch_indices = list(range(i, min(i+BATCH_SIZE, len(test_prompts))))
        
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}: prompts {i+1}-{min(i+BATCH_SIZE, len(test_prompts))}")
        for j, (idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
            print(f"  Prompt {idx+1}: '{prompt}'")
        
        # 1. Constant control (baseline)
        print("  Generating with constant control...")
        constant_audios = controller.generate(
            batch_prompts,
            layers_to_control=layers_to_control,
            control_coef=CONTROL_COEF,
            layer_weights=layer_weights,
            inject_chance=INJECT_CHANCE,
            max_new_tokens=1500
        )
        for j, (prompt_idx, audio) in enumerate(zip(batch_indices, constant_audios)):
            path = f"{OUTPUT_DIR}/prompt{prompt_idx}_constant.flac"
            sf.write(path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
        print(f"    ✓ Saved {len(batch_indices)} constant control files")
        
        # 2. Exponential decay
        print("  Generating with exponential decay...")
        decay_audios = controller.generate(
            batch_prompts,
            layers_to_control=layers_to_control,
            control_coef=CONTROL_COEF,
            time_control_fn=exponential_decay,
            layer_weights=layer_weights,
            inject_chance=INJECT_CHANCE,
            max_new_tokens=1500
        )
        for j, (prompt_idx, audio) in enumerate(zip(batch_indices, decay_audios)):
            path = f"{OUTPUT_DIR}/prompt{prompt_idx}_exponential_decay.flac"
            sf.write(path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
        print(f"    ✓ Saved {len(batch_indices)} exponential decay files")
        
        # 3. Linear ramp up
        print("  Generating with linear ramp up...")
        ramp_audios = controller.generate(
            batch_prompts,
            layers_to_control=layers_to_control,
            control_coef=CONTROL_COEF,
            time_control_fn=linear_ramp,
            layer_weights=layer_weights,
            inject_chance=INJECT_CHANCE,
            max_new_tokens=1500
        )
        for j, (prompt_idx, audio) in enumerate(zip(batch_indices, ramp_audios)):
            path = f"{OUTPUT_DIR}/prompt{prompt_idx}_linear_ramp.flac"
            sf.write(path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
        print(f"    ✓ Saved {len(batch_indices)} linear ramp files")
    
    print("\n" + "="*60)
    print("✓ Temporal control generation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60)
    print("\nConfiguration used:")
    print(f"  Concept: {CONCEPT_NAME}")
    print(f"  Layers: {LAYER_SELECTION} ({len(layers_to_control)} layers)")
    print(f"  Inject chance: {INJECT_CHANCE}")
    print(f"  Control coefficient: {CONTROL_COEF}")
    print("\nListen to how the control varies over time in each example!")

if __name__ == "__main__":
    main()
