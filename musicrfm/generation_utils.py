import torch
import random

def generate_on_text(model, tokenizer, input_text, **kwargs):
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # Generate output
    outputs = model.generate(
        **inputs,
        **kwargs,
    )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text

def hook_model(model, directions, layers_to_control, control_coef, inject_chance=1, component_idxs=[0], time_control_fn=None, layer_weights=None):
    hooks = {}
    
    # Create a time step counter for each layer
    time_step_counters = {layer_idx: 0 for layer_idx in layers_to_control}
    
    # Create layer weight map with dict comprehension
    layer_weight_map = {layer_idx: layer_weights[i] if layer_weights is not None else 1.0 
                        for i, layer_idx in enumerate(layers_to_control)}
    
    # Determine if we need random checks
    use_random = inject_chance < 1

    target_device = model.device
    target_dtype = next(model.parameters()).dtype
    
    for layer_idx in layers_to_control:
        if len(component_idxs) == 1:
            control_vec = directions[layer_idx][component_idxs[0]]
        else:
            control_vec = torch.stack([directions[layer_idx][idx] for idx in component_idxs]).sum(dim=0)
        
        if len(control_vec.shape) == 1:
            control_vec = control_vec.reshape(1, 1, -1)
            
        control_vec = control_vec.to(dtype=target_dtype, device=target_device)
        
        layer_weight = layer_weight_map[layer_idx]
        if time_control_fn is None:
            precomputed_vec = control_coef * layer_weight * control_vec
        else:
            precomputed_vec = None
               
        block = model.model.decoder.layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef, 
                      time_control_fn=time_control_fn, time_counters=time_step_counters, 
                      layer_idx=layer_idx, layer_weight=layer_weight, 
                      use_random=use_random, inject_chance=inject_chance,
                      precomputed_vec=precomputed_vec):
            """
            note that module, input are unused, but are
            required by torch.
            """ 
            
            new_output = output[0]
            
            if use_random and random.random() >= inject_chance:
                return output
            
            if precomputed_vec is not None:
                # Static case - use precomputed
                control_to_add = precomputed_vec
            else:
                # Dynamic case - compute time-varying coefficient
                current_time = time_counters[layer_idx]
                time_counters[layer_idx] += 1
                
                dynamic_control_coef = time_control_fn(current_time, control_coef)
                weighted_control_coef = dynamic_control_coef * layer_weight
                control_to_add = weighted_control_coef * control_vec
            
            new_output = new_output + control_to_add
            
            if isinstance(output, tuple):
                new_output = (new_output,) + output[1:] 
            
            return new_output
        
        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle
    
    return hooks


def multidirection_hook_model(model, directions_list, layers_to_control, control_coefs, time_control_fns=None, layer_weights=None, inject_chances=None, component_idx=0):
    hooks = {}
    
    # Create a single global time counter that increments with each generation step
    global_time_counter = 0
    
    # Flatten all layers from all concepts FIRST
    all_layers = set()
    for layers in layers_to_control:
        all_layers.update(layers)
    
    if not isinstance(control_coefs, (list, tuple)):
        control_coefs = [control_coefs] * len(directions_list)
    
    if time_control_fns is not None:
        if not isinstance(time_control_fns, (list, tuple)):
            time_control_fns = [time_control_fns] * len(directions_list)
    else:
        time_control_fns = [None] * len(directions_list)
    
    if inject_chances is not None:
        if not isinstance(inject_chances, (list, tuple)):
            inject_chances = [inject_chances] * len(directions_list)
    else:
        inject_chances = [1.0] * len(directions_list)
    
    if layer_weights is not None:
        if not isinstance(layer_weights[0], (list, tuple)):
            layer_weights = [layer_weights] * len(directions_list)
    else:
        layer_weights = []
        for _ in directions_list:
            weights = {layer_idx: 1.0 for layer_idx in all_layers}
            layer_weights.append(weights)
    
    target_device = model.device
    target_dtype = next(model.parameters()).dtype
    
    max_layer = max(all_layers)
    
    layer_direction_data = {}  # {layer_idx: [(dir_idx, control_vec, control_coef, time_fn, layer_weight, inject_chance, use_random, precomputed_vec), ...]}
    
    for layer_idx in all_layers:
        direction_data = []
        for dir_idx, directions in enumerate(directions_list):
            if layer_idx in layers_to_control[dir_idx]:
                control_vec = directions[layer_idx][component_idx]
                if len(control_vec.shape) == 1:
                    control_vec = control_vec.reshape(1, 1, -1)
                
                control_vec = control_vec.to(dtype=target_dtype, device=target_device)
                
                control_coef = control_coefs[dir_idx]
                time_control_fn = time_control_fns[dir_idx]
                layer_weight = layer_weights[dir_idx][layer_idx]
                inject_chance = inject_chances[dir_idx]
                use_random = inject_chance < 1.0
                
                if time_control_fn is None:
                    precomputed_vec = control_coef * layer_weight * control_vec
                else:
                    precomputed_vec = None
                
                direction_data.append((dir_idx, control_vec, control_coef, time_control_fn, 
                                      layer_weight, inject_chance, use_random, precomputed_vec))
        
        layer_direction_data[layer_idx] = direction_data
    
    for layer_idx in all_layers:
        block = model.model.decoder.layers[layer_idx]
        direction_data = layer_direction_data[layer_idx]

        def block_hook(module, input, output, layer_idx=layer_idx, direction_data=direction_data, 
                      max_layer=max_layer):
            nonlocal global_time_counter
            
            combined_control_vec = None
            
            for dir_idx, control_vec, control_coef, time_control_fn, layer_weight, inject_chance, use_random, precomputed_vec in direction_data:
                if use_random and random.random() >= inject_chance:
                    continue
                
                if precomputed_vec is not None:
                    # Static case - use precomputed
                    control_to_add = precomputed_vec
                else:
                    # Dynamic case - compute time-varying coefficient
                    dynamic_control_coef = time_control_fn(global_time_counter, control_coef)
                    weighted_control_coef = dynamic_control_coef * layer_weight
                    control_to_add = weighted_control_coef * control_vec
                
                # Accumulate control vectors
                if combined_control_vec is None:
                    combined_control_vec = control_to_add
                else:
                    combined_control_vec = combined_control_vec + control_to_add
            
            # Only increment time counter for the LAST layer (indicating completion of a generation step)
            if layer_idx == max_layer:
                global_time_counter += 1
            
            if combined_control_vec is not None:
                new_output = output[0]
                new_output = new_output + combined_control_vec
                
                if isinstance(output, tuple):
                    new_output = (new_output,) + output[1:] 
                
                return new_output
            else:
                return output
        
        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle
    
    return hooks


def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()