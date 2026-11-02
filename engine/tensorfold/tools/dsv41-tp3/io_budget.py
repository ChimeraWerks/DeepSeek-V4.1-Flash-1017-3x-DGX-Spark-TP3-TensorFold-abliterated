"""Bound preparation reads without changing global cache or scheduler state.

NOREUSE is a Linux reclaim hint, not a memory limit. No DONTNEED/drop_caches:
source pages may also be in use by production. Use a shared Budget per process.
"""
import math
import os
import time


class Budget:
    def __init__(self, mib_s=0, *, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(mib_s) or mib_s < 0:
            raise ValueError('Read rate must be finite and nonnegative')
        self.rate=mib_s*(1<<20)
        self.clock=clock; self.sleep=sleep
        self.deadline=clock()
        self.pending=0

    def before_read(self, size):
        if size < 0: raise ValueError('Bounded reads require an explicit size')
        if self.rate and size:
            self.pending+=size
            # Packed tensor slices can be tiny. Bound burst allowance to 64KiB
            # instead of a clock syscall and micro-sleep per codeword row.
            if self.pending < 65536: return
            now=self.clock()
            # Idle time must not accumulate an unrestricted burst allowance.
            self.deadline=max(now,self.deadline)+self.pending/self.rate
            self.pending=0
            self.sleep(max(0,self.deadline-now))


class Reader:
    def __init__(self, file, budget):
        self.file=file; self.budget=budget
        if hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_NOREUSE'):
            try: os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_NOREUSE)
            except OSError: pass  # Hint may be unsupported by the filesystem.

    def read(self, size):
        self.budget.before_read(size)
        return self.file.read(size)

    def seek(self, *args): return self.file.seek(*args)
    def close(self): self.file.close()
    def __enter__(self): return self
    def __exit__(self, *unused): self.close()
