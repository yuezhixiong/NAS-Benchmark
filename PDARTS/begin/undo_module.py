import torch
from collections import OrderedDict

state_dict = torch.load('weights.pt')

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

torch.save(new_state_dict, 'weights_unmodule.pt')