"""
Example: Multi-direction control (controlling multiple concepts simultaneously)

This example demonstrates how to control multiple musical attributes
at the same time, such as both the note and the tempo.

Includes all advanced options from 02_generate_controlled.py plus:
- Separate layer selection for each concept
- Separate time control functions for each concept
- Separate inject chances for each concept
- Control coefficient combinations
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

def main():
    # ==================== Configuration ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Two concepts to control simultaneously
    CONCEPT_NAMES = ["note_C_ncomp12", "note_D#_ncomp12"]
    DIRECTIONS_PATH = "./trained_concepts/directions"
    RESULTS_PATH = "./trained_concepts/results"
    OUTPUT_DIR = "./trained_concepts/generations_multidir"
    BATCH_SIZE = 4
    
    test_prompts = [
        "A relaxing jazz song with piano",
        "Upbeat electronic dance music with synths",
        "Slow classical piano piece",
        "Fast-paced rock song with guitar"
    ]
    
    # ==================== Advanced Options ====================
    
    # Layer selection method (choose one) - APPLIES TO ALL CONCEPTS
    LAYER_SELECTION = "exp_weighting"  # Options: "all", "top_k", "exp_weighting"
    TOP_K = 16  # Number of top layers (if using "top_k")
    EXP_BASE_WEIGHT = 1.0  # Base weight for exponential dropout
    EXP_DECAY_RATE = 0.95  # Decay rate for exponential dropout
    
    # Control coefficient combinations to test
    # Each tuple is (coef1, coef2) for (concept1, concept2)
    CONTROL_COMBINATIONS = [
        (0.5, 0.5),   # Both equally
        (0.7, 0.3),   # More concept 1
        (0.3, 0.7),   # More concept 2
    ]
    
    # Time control functions (one per concept, None for constant)
    # Set to None for constant, or use a function like exponential_decay
    TIME_CONTROL_FNS = [None, None]  # Example: [exponential_decay, None]
    
    # Probabilistic injection (one per concept)
    INJECT_CHANCES = [0.3, 0.3]  # Probability of applying each concept's control
    
    # Regression flags (one per concept)
    IS_REGRESSION = [False, False]  # Set to True for tempo/continuous concepts
    
    # ==================== Validate Paths and Load Results ====================
    
    print(f"Multi-Direction Control Example")
    print(f"Concepts: {CONCEPT_NAMES}")
    print(f"Layer selection: {LAYER_SELECTION}")
    print(f"Inject chances: {INJECT_CHANCES}")
    
    # Check if all concepts exist
    for concept_name in CONCEPT_NAMES:
        concept_dir_path = os.path.join(DIRECTIONS_PATH, f"music_rfm_{concept_name}_musicgen_large.pkl")
        if not os.path.exists(concept_dir_path):
            print(f"\n✗ Error: Concept {concept_name} not found at {DIRECTIONS_PATH}")
            print(f"   Looking for: music_rfm_{concept_name}_musicgen_large.pkl")
            print("\nPlease run 01_train_note_direction.py to train all concepts.")
            return
    
    # Load results files if needed for layer selection
    results_list = []
    if LAYER_SELECTION in ["top_k", "exp_weighting"]:
        for concept_name in CONCEPT_NAMES:
            results_file = os.path.join(RESULTS_PATH, f"{concept_name}.json")
            if not os.path.exists(results_file):
                print(f"\n✗ Error: Results file required for '{LAYER_SELECTION}' layer selection")
                print(f"   Looking for: {results_file}")
                print("\nOptions:")
                print("  1. Change LAYER_SELECTION to 'all' (doesn't require results file)")
                print("  2. Re-train the concepts and ensure results are saved")
                return
            
            print(f"\nLoading results from: {results_file}")
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                results_list.append(results)
                print(f"✓ Results loaded for {concept_name}")
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
    
    # ==================== Load Concepts and Select Layers ====================
    
    directions_list = []
    layers_to_control_list = []
    layer_weights_list = []
    
    for i, concept_name in enumerate(CONCEPT_NAMES):
        print(f"\nLoading concept {i+1}/{len(CONCEPT_NAMES)}: {concept_name}")
        
        controller.load(
            concept=concept_name,
            model_name="musicgen_large",
            path=DIRECTIONS_PATH
        )
        
        # Store this concept's directions
        directions_list.append(controller.directions.copy())
        print(f"  ✓ Loaded concept {i+1}")
        
        # Select layers for this concept
        layer_weights = None
        
        if LAYER_SELECTION == "all":
            layers_to_control = list(range(-1, -48, -1))
            print(f"  Using all {len(layers_to_control)} layers")
            
        elif LAYER_SELECTION == "top_k":
            layers_to_control = select_top_k_layers(results_list[i], TOP_K, IS_REGRESSION[i])
                
        elif LAYER_SELECTION == "exp_weighting":
            layers_to_control, layer_weights = select_exponential_layer_dropout(
                results_list[i], IS_REGRESSION[i], EXP_BASE_WEIGHT, EXP_DECAY_RATE
            )
        else:
            raise ValueError(f"Invalid layer selection method: {LAYER_SELECTION}")
        
        layers_to_control_list.append(layers_to_control)
        layer_weights_list.append(layer_weights)
    
    # ==================== Generate Music ====================
    
    sampling_rate = music_model.config.audio_encoder.sampling_rate
    
    print("\n" + "="*60)
    print("Generating with Multi-Direction Control")
    print("="*60)
    
    # Process prompts in batches
    for i in range(0, len(test_prompts), BATCH_SIZE):
        batch_prompts = test_prompts[i:i+BATCH_SIZE]
        batch_indices = list(range(i, min(i+BATCH_SIZE, len(test_prompts))))
        
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}: prompts {i+1}-{min(i+BATCH_SIZE, len(test_prompts))}")
        for j, (idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
            print(f"  Prompt {idx+1}: '{prompt}'")
        
        for coef1, coef2 in CONTROL_COMBINATIONS:
            control_coefs = [coef1, coef2]
            print(f"  Generating with control={control_coefs}...")
            
            # Apply multi-direction control
            controlled_audios = controller.multidirection_generate(
                batch_prompts,
                directions_list=directions_list,
                layers_to_control=layers_to_control_list,
                control_coefs=control_coefs,
                time_control_fns=TIME_CONTROL_FNS,
                layer_weights=layer_weights_list,
                inject_chances=INJECT_CHANCES,
                max_new_tokens=1500
            )
            
            # Save each audio in the batch
            for j, (prompt_idx, audio) in enumerate(zip(batch_indices, controlled_audios)):
                path = f"{OUTPUT_DIR}/prompt{prompt_idx}_c1_{coef1}_c2_{coef2}.flac"
                sf.write(path, audio[0].cpu().numpy(), sampling_rate, format='FLAC')
            print(f"    ✓ Saved {len(batch_indices)} files for control={control_coefs}")
    
    print("\n" + "="*60)
    print("✓ Multi-direction generation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60)
    print("\nConfiguration used:")
    print(f"  Concepts: {CONCEPT_NAMES}")
    print(f"  Layers: {LAYER_SELECTION}")
    print(f"  Inject chances: {INJECT_CHANCES}")
    print(f"  Time control: {['Yes' if fn else 'No' for fn in TIME_CONTROL_FNS]}")
    print("\nCompare the different combinations to hear how concepts interact!")

if __name__ == "__main__":
    main()
