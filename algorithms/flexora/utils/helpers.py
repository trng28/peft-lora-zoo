import re

def get_layer_index_from_name(module_name):
    """
    "base_model.model.model.layers.15.self_attn.q_proj" -> 15
    """
    # Pattern looks for 'layers.X.' or 'h.X.' depending on architecture
    match = re.search(r'\.(layers|h|block)\.(\d+)\.', module_name)
    if match:
        return int(match.group(2))
    return None



def get_specific_target_modules(selected_layer_indices, target_suffixes):
    """
    Generates specific module names for Stage 2 Fine-tuning.
    Compatible with Llama-3.2 structure.
    """
    targets = []
    selected_layer_indices = [int(i) for i in selected_layer_indices]
    
    for i in selected_layer_indices:
        for suffix in target_suffixes:
            # Llama-3.2:
            # Attention: model.layers.{i}.self_attn.{q_proj}
            # MLP:       model.layers.{i}.mlp.{gate_proj}
            
            if suffix in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                targets.append(f"layers.{i}.self_attn.{suffix}")
            elif suffix in ["gate_proj", "up_proj", "down_proj"]:
                targets.append(f"layers.{i}.mlp.{suffix}")
            else:
                targets.append(f"layers.{i}.{suffix}")
                
    return targets



# def get_specific_target_modules(selected_layer_indices, target_suffixes):
#     """
#     Generates the list of specific module names for Stage 2 Fine-tuning.
#     """
#     # Format for Llama: "model.layers.{i}.self_attn.{suffix}"
#     targets = []
#     for i in selected_layer_indices:
#         for suffix in target_suffixes:
#             targets.append(f"layers\.{i}\..*{suffix}")
#     return targets


