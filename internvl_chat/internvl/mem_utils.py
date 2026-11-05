# A set of utils for memory monitoring.
# These are all taken and adapted from this amazing
# repo: https://github.com/ppwwyyxx/RAM-multiprocess-dataloader
import torch
import psutil
from typing import Optional, Dict, List
from collections import defaultdict
from tabulate import tabulate


def to_gb(size: int) -> float:
    return size / (1024 ** 3)


def get_tensor_size(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    res['fds'] = len(psutil.Process(pid).open_files())
    for mmap in psutil.Process(pid).memory_maps():
        res['rss'] += mmap.rss
        res['pss'] += mmap.pss
        res['uss'] += mmap.private_clean + mmap.private_dirty
        res['shared'] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith('/'):
            res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
    return res


def format_mem_info(
    pid: int, 
    tensors: Optional[List[Dict[str, torch.Tensor]]] = None,
    decimals: int = 2
) -> str:
    mem_info = get_mem_info(pid)
    headers, line = [], []
    for k, v in mem_info.items():
        if k == 'fds':
            line.append(f"{v:,}")
        else:
            line.append(f"{round(to_gb(v), decimals)}G")
        headers.append(k)
    if tensors is not None and len(tensors) > 0:
        first = tensors[0]
        tensors_size = []
        tensors_headers = []
        for k, v in first.items():
            size = sum(get_tensor_size(t[k]) for t in tensors)
            tensors_size.append(size)
            tensors_headers.append(k)
        line.extend([f"{round(to_gb(s), decimals)}G" for s in tensors_size])
        headers.extend(tensors_headers)
    return tabulate([line], headers=headers, tablefmt="plain")
