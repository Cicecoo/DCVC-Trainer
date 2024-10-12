import torch

def load_submodule_params(submodule, whole_module_checkpoint, submodule_name):
    submodule_params = submodule.state_dict()
    whole_module_state_dict = torch.load(whole_module_checkpoint)

    # print("submodule_params", submodule_params.keys())
    # print("whole_module_state_dict", whole_module_state_dict.keys())

    for name, param in submodule_params.items():
        full_name = submodule_name + '.' + name
        print("loading", full_name, "to", name)
        if full_name in whole_module_state_dict:
            submodule_params[name] = whole_module_state_dict[full_name]
        else:
            print("WARNING: could not find", full_name, "for", name, "in checkpoint")

    submodule.load_state_dict(submodule_params)


def freeze_submodule(submodule_list):
    for submodule in submodule_list:
        for param in submodule.parameters():
            param.requires_grad = False

def unfreeze_submodule(submodule_list):
    for submodule in submodule_list:
        for param in submodule.parameters():
            param.requires_grad = True