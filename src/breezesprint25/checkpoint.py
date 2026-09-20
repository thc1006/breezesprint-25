"""Verified multi-file SafeTensors view. Never load Pickle or missing random weights."""
from contextlib import ExitStack
import json
from pathlib import Path
from safetensors import safe_open


class TensorStore:
    def __init__(self, filenames, index_path=None):
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]
        self.paths = [Path(p) for p in filenames]
        if not self.paths or len({p.name for p in self.paths}) != len(self.paths):
            raise ValueError('Empty or duplicate checkpoint shards.')
        self.index_path = Path(index_path) if index_path else None
        self.stack = ExitStack()
        self.owners = {}

    def __enter__(self):
        try:
            actual_map = {}
            for path in self.paths:
                handle = self.stack.enter_context(safe_open(str(path), framework='np'))
                for name in handle.keys():
                    if name in self.owners:
                        raise ValueError('Duplicate tensor across shards: ' + name)
                    self.owners[name] = handle
                    actual_map[name] = path.name
            if self.index_path:
                raw = json.loads(self.index_path.read_text(encoding='utf-8'))
                declared = raw.get('weight_map')
                if not isinstance(declared, dict) or declared != actual_map:
                    raise ValueError('Checkpoint index disagrees with shard tensor ownership.')
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def keys(self): return self.owners.keys()
    def get_slice(self, name): return self.owners[name].get_slice(name)
    def get_tensor(self, name): return self.owners[name].get_tensor(name)
