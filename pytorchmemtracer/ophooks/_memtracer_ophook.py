import torch
from . import BaseOpHook
from concurrent.futures import ThreadPoolExecutor
from time import sleep, time
import psutil
import pickle

def get_cuda_memory_used(device):
    """
    Get the free memory info of device.
    Notice that for CPU, this function will return 1/N of the total free memory,
    where N is the world size.
    """
    ret = torch.cuda.memory_allocated()
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
    return ret


class AsyncMemoryMonitor:
    def __init__(self, power=10):
        """
        An Async Mem Monitor runing during computing.
        Sampling GPU memory usage of the current GPU dev
        at interval of 1/(10**power) sec.
        """
        self.keep_measuring = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.monitor_thread = None
        self.interval = 1 / (10 ** power)
        self.time_stamps = []
        self.mem_stats = []

    def set_interval(self, power: int):
        self.interval = 1 / (10 ** power)

    def is_measuring(self):
        return self.keep_measuring
    
    def start(self):
        self.keep_measuring = True
        self.monitor_thread = self.executor.submit(self._measure_usage)
        
    def finish(self):
        if self.keep_measuring is False:
            return 0
        self.keep_measuring = False
        max_usage = self.monitor_thread.result()
        self.monitor_thread = None
        self.time_stamps.append(time())
        self.mem_stats.append(max_usage)
        return max_usage

    def _measure_usage(self):
        max_usage = 0
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                get_cuda_memory_used(dev),
            )
            sleep(self.interval)
        return max_usage

    def state_dict(self):
        return {
            "time_stamps" : self.time_stamps,
            "mem_stats" : self.mem_stats,
        }

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.state_dict(), f)
    
class MemTracerOpHook(BaseOpHook):
    def __init__(self):
        super().__init__()
        self.async_mem_monitor = AsyncMemoryMonitor()

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            if self.async_mem_monitor.is_measuring():
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
            if self.async_mem_monitor.is_measuring():
                self.async_mem_monitor.finish()
            self.async_mem_monitor.start()
            # print(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            if self.async_mem_monitor.is_measuring():
                self.async_mem_monitor.finish()
            # print(f'BWD POST {module.__class__.__name__}')
        
    def post_iter(self):
        if self.async_mem_monitor.is_measuring():
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