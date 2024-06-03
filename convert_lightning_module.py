
r"""
Read the lightning checkpoint and save weight only.

Useage:
Show the stricture of checkpoint
```
python convert_lightning_module.py --ckpt_path=path/to/checkpoint.ckpt -v
```

remove the prefix of the checkpoint
```
python convert_lightning_module.py --ckpt_path=path/to/checkpoint.ckpt -x model --out_path=path/to/output.pt
```
"""

import argparse

import torch

from basic_config_setting import *


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}

    def add_child(self, child_name):
        if child_name in self.children:
            return self.children[child_name]
        else:
            new_child = TreeNode(child_name)
            self.children[child_name] = new_child
            return new_child

    def build_tree(self, path):
        current_node = self
        for node_name in path.split('.'):
            current_node = current_node.add_child(node_name)

    def is_one_child(self):
        return len(self.children) == 1

    def get_one_child(self):
        return list(self.children.values())[0]

    def __str__(self, indent=0, max_indent=-1):
        if max_indent > 0 and indent > max_indent:
            return ''
        result = '  ' * indent + self.name + '\n'
        for child in self.children.values():
            result += child.__str__(indent + 1, max_indent)
        return result


_args = argparse.ArgumentParser(
    __file__, description="Read the lightning checkpoint and save weight only.")
_args.add_argument("--ckpt_path", type=str, required=True,
                   help="The path of the checkpoint file.")
_args.add_argument("--out_path", type=str, default=None,
                   help="The path of the output file.")
_args.add_argument("-x", "--remove_prefix", type=str, default="",
                   help="Remove the model prefix in the weight name.")
_args.add_argument("-v", "--show_prefix_depth", type=int, default=0,
                   help="Show the depth of the model prefix in the weight name.")
args = _args.parse_args()

lightning_checkpoint = torch.load(
    args.ckpt_path, map_location=torch.device('cpu'))
try:
    state_dict = lightning_checkpoint['state_dict']  # type: dict
    print(lightning_checkpoint.keys())
except Exception:
    state_dict = lightning_checkpoint

print("loaded state dict")

if args.show_prefix_depth > 0:
    print("show prefix")
    # all key has the format a.b.c.c
    # find all commend
    mkey = TreeNode("")
    for k in state_dict.keys():
        mkey.build_tree(k)

    print(mkey.__str__(0, args.show_prefix_depth))

if args.out_path is None:
    print("out_path is None")
    print("do not save the state_dict")
    print("done")
    exit(0)


if args.remove_prefix != "":
    remove_prefix = args.remove_prefix  # type: str
    if not remove_prefix.endswith("."):
        remove_prefix += "."

    print("remove prefix \033[32m", remove_prefix, "\033[0m")
    klen = len(remove_prefix)
    all_keys = list(state_dict.keys())  # type: list[str]
    for k in all_keys:
        old_k = k
        if k.startswith(remove_prefix):
            k = k[klen:]
        else:
            continue
        print(f"{old_k} => {k}")
        state_dict[k] = state_dict.pop(old_k)

print("save new state dict")
torch.save(state_dict, args.out_path)
print("done")
