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
    
    # Create a mapping from layer_idx to its weight
    layer_weight_map = {}
    if layer_weights is not None:
        for i, layer_idx in enumerate(layers_to_control):
            layer_weight_map[layer_idx] = layer_weights[i]
    else:
        # If no weights provided, use uniform weights of 1.0
        for layer_idx in layers_to_control:
            layer_weight_map[layer_idx] = 1.0
    
    for layer_idx in layers_to_control:
        control_vec = directions[layer_idx][component_idxs[0]]
        for component_idx in component_idxs[1:]:
            control_vec += directions[layer_idx][component_idx]
        if len(control_vec.shape)==1:
            control_vec = control_vec.reshape(1,1,-1)
               
        block = model.model.decoder.layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef, 
                      time_control_fn=time_control_fn, time_counters=time_step_counters, 
                      layer_idx=layer_idx, layer_weight=layer_weight_map[layer_idx]):
            """
            note that module, input are unused, but are
            required by torch.
            """ 
            
            new_output = output[0]
            
            # Calculate time-varying control coefficient
            if time_control_fn is not None:
                # Get current time step and increment
                current_time = time_counters[layer_idx]
                time_counters[layer_idx] += 1
                
                # Apply time control function to get dynamic control coefficient
                dynamic_control_coef = time_control_fn(current_time, control_coef)
                #print(f"Layer {layer_idx}, Time step {current_time}: Dynamic control coefficient: {dynamic_control_coef}")
            else:
                # Use static control coefficient
                dynamic_control_coef = control_coef

            # Apply layer weight as a multiplier to the control coefficient
            weighted_control_coef = dynamic_control_coef * layer_weight
            
            # Only add the control vector with probability inject_chance
            if inject_chance >= 1 or random.random() < inject_chance:
                new_output = new_output + weighted_control_coef * control_vec.to(dtype=new_output.dtype, device=new_output.device)
            
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
    
    # Process each layer that appears in any concept
    for layer_idx in all_layers:
        block = model.model.decoder.layers[layer_idx]

        def block_hook(module, input, output, layer_idx=layer_idx):
            nonlocal global_time_counter
            
            combined_control_vec = None
            
            for dir_idx, directions in enumerate(directions_list):
                # Only process this direction if it has this layer
                if layer_idx in layers_to_control[dir_idx]:
                    control_vec = directions[layer_idx][component_idx]
                    if len(control_vec.shape) == 1:
                        control_vec = control_vec.reshape(1, 1, -1)
                    
                    control_coef = control_coefs[dir_idx]
                    time_control_fn = time_control_fns[dir_idx]
                    layer_weight = layer_weights[dir_idx][layer_idx]
                    inject_chance = inject_chances[dir_idx]
                    
                    # Use the global time counter for time control
                    dynamic_control_coef = time_control_fn(global_time_counter, control_coef) if time_control_fn is not None else control_coef
                    
                    weighted_control_coef = dynamic_control_coef * layer_weight
                    
                    # Only add this direction's control vector with probability inject_chance
                    if inject_chance >= 1.0 or random.random() < inject_chance:
                        if combined_control_vec is None:
                            combined_control_vec = weighted_control_coef * control_vec
                        else:
                            combined_control_vec = combined_control_vec + weighted_control_coef * control_vec
            
            # Only increment time counter for the LAST layer (indicating completion of a generation step)
            if layer_idx == max(all_layers):
                global_time_counter += 1
            
            if combined_control_vec is not None:
                new_output = output[0]
                new_output = new_output + combined_control_vec.to(dtype=new_output.dtype, device=new_output.device)
                
                if isinstance(output, tuple):
                    new_output = (new_output,) + output[1:] 
                
                return new_output
            else:
                return output
        
        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle
    
    return hooks


# def hook_model(model, directions, layers_to_control, control_coef, component_idx=0, time_control_fn=None):
#     hooks = {}
#     for layer_idx in layers_to_control:
#         control_vec = directions[layer_idx][component_idx]
#         if len(control_vec.shape)==1:
#             control_vec = control_vec.reshape(1,1,-1)
               
               
#         block = model.model.decoder.layers[layer_idx]

#         def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
#             """
#             note that module, input are unused, but are
#             required by torch.
#             """ 
            
#             new_output = output[0]

#             new_output = new_output + control_coef*control_vec.to(dtype=new_output.dtype, device=new_output.device)
            
#             if isinstance(output, tuple):
#                 new_output = (new_output,) + output[1:] 
            
#             return new_output
        
#         hook_handle = block.register_forward_hook(block_hook)
#         hooks[layer_idx] = hook_handle
    
#     return hooks


def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()