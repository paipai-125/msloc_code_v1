import os
import sys

# Adjust sys.path so that the `trace` package (located at MSLoc/Trace/trace) is importable.
current_file_path = os.path.abspath(__file__)
trace_pkg_dir = os.path.dirname(current_file_path)  # .../trace
repo_root = os.path.dirname(trace_pkg_dir)          # .../Trace (MSLoc/Trace)

if repo_root not in sys.path:
    sys.path.append(repo_root)
if trace_pkg_dir not in sys.path:
    sys.path.append(trace_pkg_dir)

# from trace.mistral_npu_monkey_patch import (
#     replace_with_torch_npu_flash_attention,
#     replace_with_torch_npu_rmsnorm
# )

# replace_with_torch_npu_flash_attention()
# replace_with_torch_npu_rmsnorm()

from trace.train_mt import train
import torch_npu
from torch_npu.contrib import transfer_to_npu

if __name__ == "__main__":
    train()