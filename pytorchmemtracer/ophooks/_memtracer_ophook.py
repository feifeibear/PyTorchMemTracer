import torch
from . import BaseOpHook
from time import time
import pickle
from abc import abstractmethod

class MemoryMonitor:
    """Base class for all types of memory monitor.
    All monitors should have a list called `time_stamps` and a list called `mem_stats`.
    """

    def __init__(self):
        self.time_stamps = []
        self.mem_stats = []

    def __len__(self):
        return len(self.mem_stats)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    def state_dict(self):
        return {
            "time_stamps": self.time_stamps,
            "mem_stats": self.mem_stats,
        }
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.state_dict(), f)

class SyncCudaMemoryMonitor(MemoryMonitor):
    """
    A synchronized cuda memory monitor.
    It only record the maximum allocated cuda memory from start point to finish point.
    """

    def __init__(self, power: int = 10):
        super().__init__()

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def finish(self):
        torch.cuda.synchronize()
        self.time_stamps.append(time())
        max_usage = torch.cuda.max_memory_allocated()
        self.mem_stats.append(max_usage)
        return max_usage

class MemTracerOpHook(BaseOpHook):
    def __init__(self):
        super().__init__()
        self.async_mem_monitor = SyncCudaMemoryMonitor()

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()
            # print(f'FWD PRE {module.__class__.__name__}')

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            self.async_mem_monitor.finish()
            # print(f'FWD POST {module.__class__.__name__}')

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()
            # print(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            self.async_mem_monitor.finish()
            # print(f'BWD POST {module.__class__.__name__}')
    
    def pre_iter(self):
        pass

    def post_iter(self):
        self.async_mem_monitor.finish()
        # print(f'post_iter')

    def save_results(self, filename):
        self.async_mem_monitor.save(filename)

    def show_mem_stats(self):
        start_timestamp = min(self.async_mem_monitor.time_stamps)
        self.async_mem_monitor.time_stamps = [elem - start_timestamp for elem in self.async_mem_monitor.time_stamps]
        min_mem_used = min(self.async_mem_monitor.mem_stats)
        self.async_mem_monitor.mem_stats = [elem - min_mem_used for elem in self.async_mem_monitor.mem_stats]
        print(self.async_mem_monitor.time_stamps)
        print(self.async_mem_monitor.mem_stats)