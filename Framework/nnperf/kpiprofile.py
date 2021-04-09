import functools
import warnings
from collections import OrderedDict, defaultdict, namedtuple

import torch.autograd.profiler as torch_profiler
import csv

from .display import traces_to_display

Trace = namedtuple("Trace", ["path", "leaf", "module"])
KPIObject = namedtuple(
    "KPIObject",
    [  # when attr value is None, profiler unsupported
        "model",
        "name",
        "self_cpu_total",
        "cpu_total",
        "self_cuda_total",
        "cuda_total",
        "self_cpu_memory",
        "cpu_memory",
        "self_cuda_memory",
        "cuda_memory",
        "occurrences",
    ],
)


def walk_modules(module, name="", path=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    yield Trace(path, len(named_children) == 0, module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class Profile(object):
    """Layer by layer profiling of PyTorch models, using the PyTorch autograd profiler."""

    def __init__(
        self, model, enabled=True, use_cuda=False, profile_memory=False, paths=None
    ):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory
        self.paths = paths
        self.entered = False
        self.exited = False
        self.traces = ()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("Profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __str__(self):
        return self.display()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                try:
                    with torch_profiler.profile(
                        use_cuda=self.use_cuda, profile_memory=self.profile_memory
                    ) as prof:
                        res = _forward(*args, **kwargs)
                except TypeError:
                    if self.profile_memory:
                        warnings.warn(
                            "`profile_memory` is unsupported in torch < 1.6",
                            RuntimeWarning,
                        )
                        self.profile_memory = False
                    with torch_profiler.profile(use_cuda=self.use_cuda) as prof:
                        res = _forward(*args, **kwargs)

                event_list = prof.function_events
                
                
                # PyTorch up until version 1.7 exposes this method. From PyTorch 1.8 onwards, 
                # it is called via EventList._build_tree at the end of the context manager.
                if hasattr(event_list, "populate_cpu_children"):
                    event_list.populate_cpu_children()
                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            module.forward = self._forwards[path]

    def raw(self):
        if self.exited:
            return (self.traces, self.trace_profile_events)

    def display(self, show_events=False):
        if self.exited:
            return traces_to_display(
                self.traces,
                self.trace_profile_events,
                show_events=show_events,
                paths=self.paths,
                use_cuda=self.use_cuda,
                profile_memory=self.profile_memory,
            )
        return "<unfinished profile>"
    
    def getKPIData(self):
        
        layers = []
        rows = []
        for trace in self.traces:
            [path, leaf, module] = trace
            current_layers = layers
            # unwrap all of the events, in case model is called multiple times
            events = [te for t_events in self.trace_profile_events[path] for te in t_events]
            for depth, name in enumerate(path, 1):
                if name not in current_layers:
                    current_layers.append(name)
                if depth == len(path) and (
                    (self.paths is None and leaf) or (self.paths is not None and path in self.paths)
                ):
                    
                    self_cpu_memory = None
                    has_self_cpu_memory = any(hasattr(e, "self_cpu_memory_usage") for e in events)
                    if has_self_cpu_memory:
                        self_cpu_memory = sum([getattr(e, "self_cpu_memory_usage", 0) for e in events])
                    cpu_memory = None
                    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
                    if has_cpu_memory:
                        cpu_memory = sum([getattr(e, "cpu_memory_usage", 0) for e in events])
                    self_cuda_memory = None
                    has_self_cuda_memory = any(hasattr(e, "self_cuda_memory_usage") for e in events)
                    if has_self_cuda_memory:
                        self_cuda_memory = sum(
                            [getattr(e, "self_cuda_memory_usage", 0) for e in events]
                        )
                    cuda_memory = None
                    has_cuda_memory = any(hasattr(e, "cuda_memory_usage") for e in events)
                    if has_cuda_memory:
                        cuda_memory = sum([getattr(e, "cuda_memory_usage", 0) for e in events])
                
                    # self CUDA time supported in torch >= 1.7
                    self_cuda_total = None
                    has_self_cuda_time = any(hasattr(e, "self_cuda_time_total") for e in events)
                    if has_self_cuda_time:
                        self_cuda_total = sum([getattr(e, "self_cuda_time_total", 0) for e in events])
                
                    kpiObject = self.format_measurements(current_layers[0], name, sum([e.self_cpu_time_total for e in events]),
                                             sum([e.cpu_time_total for e in events]), self_cuda_total,
                            sum([e.cuda_time_total for e in events]), self_cpu_memory, cpu_memory, 
                            self_cuda_memory, cuda_memory, len(self.trace_profile_events[path]))
                    
                    rows.append(kpiObject)
        self.exportToCSV(rows)
    
    def format_measurements(self, model, name, self_cpu_total, cpu_total, self_cuda_total,
                            cuda_total, self_cpu_memory, cpu_memory, self_cuda_memory,
                            cuda_memory, occurrences):

        self_cpu_total = self_cpu_total/1000.0
        cpu_total = cpu_total/1000.0
        self_cuda_total = self_cuda_total/1000.0 if self_cuda_total is not None else 0
        cuda_total = cuda_total/1000.0 if cuda_total else 0
        self_cpu_memory = (
            self_cpu_memory/1024.0
            if self_cpu_memory is not None
            else 0
        )
        cpu_memory = (
            cpu_memory/1024.0
            if cpu_memory is not None
            else 0
        )
        self_cuda_memory = (
            self_cuda_memory/1024.0
            if self_cuda_memory is not None
            else 0
        )
        cuda_memory = (
            cuda_memory/1024.0
            if cuda_memory is not None
            else 0
        )
        occurrences = occurrences if occurrences else 0
        
        return KPIObject(
        model = model,
        name = name,
        self_cpu_total=self_cpu_total,
        cpu_total=cpu_total,
        self_cuda_total=self_cuda_total,
        cuda_total=cuda_total,
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )
        
    def exportToCSV(self, rows):
        model = rows[0].model
        file = model + "KPI.csv"
        f = open(file, 'w')
        with f:
            headers = ['MODULE', 'SELF_CPU_TOTAL', 'SELF_CPU_TIME_UOM', 'CPU_TOTAL', 'CPU_TOTAL_UOM', 
                       'SELF_GPU_TOTAL', 'SELF_GPU_UOM', 'GPU_TOTAL', 'GPU_TOTAL_UOM',
                       'SELF_CPU MEM','SELF_CPU_MEM_UOM', 'CPU_MEM','CPU_MEM_UOM','SELF_GPU_MEM',
                       'SELF_GPU_MEM_UOM','GPU_MEM','GPU_MEM_UOM','NUMBER_OF_CALLS']
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for i in range(len(rows)):
                kpi = rows[i]
                writer.writerow({'MODULE':kpi.name, 
                                 'SELF_CPU_TOTAL':kpi.self_cpu_total, 
                                 'SELF_CPU_TIME_UOM': 'ms',
                                 'CPU_TOTAL':kpi.cpu_total, 
                                 'CPU_TOTAL_UOM': 'ms', 
                                 'SELF_GPU_TOTAL':kpi.self_cuda_total, 
                                 'SELF_GPU_UOM':'ms',
                                 'GPU_TOTAL':kpi.cuda_total, 
                                 'GPU_TOTAL_UOM': 'ms',
                                 'SELF_CPU MEM':kpi.self_cpu_memory, 
                                 'SELF_CPU_MEM_UOM': 'kb',
                                 'CPU_MEM':kpi.cpu_memory,
                                 'CPU_MEM_UOM': 'kb', 
                                 'SELF_GPU_MEM':kpi.self_cuda_memory, 
                                 'SELF_GPU_MEM_UOM':'kb',
                                 'GPU_MEM':kpi.cuda_memory,
                                 'GPU_MEM_UOM': 'kb',
                                 'NUMBER_OF_CALLS': kpi.occurrences})
        