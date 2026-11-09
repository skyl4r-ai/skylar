# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Per-node GPU telemetry for long runs: power, utilisation, memory and the energy counter of every GPU
on the node, sampled by a background thread into `<out>/telemetry/<host>.jsonl`.

One process per node runs it (the local main). The energy comes from NVML's cumulative counter
(`nvmlDeviceGetTotalEnergyConsumption`, millijoules since the driver loaded, Volta and newer): the
difference between the first and the last sample is the energy of the run on that GPU, without
integrating power by hand. Where the counter is missing the thread integrates the power samples.

NVML is called through ctypes on the driver's own `libnvidia-ml.so.1`, so there is no extra package.
Any NVML failure turns telemetry off with one message; it never stops training.

    from training.telemetry import NodeTelemetry
    tel = NodeTelemetry(out_dir, every_s=30).start()
    ...
    summary = tel.stop()          # {"energy_kwh": ..., "gpu_hours": ..., ...}, also written to the file
"""
import ctypes
import json
import socket
import threading
import time
from pathlib import Path


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class NVML:
    """The few NVML calls the telemetry needs. Raises OSError/RuntimeError if NVML is unusable."""

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit failed")
        n = ctypes.c_uint()
        if self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0:
            raise RuntimeError("nvmlDeviceGetCount failed")
        self.handles = []
        for i in range(n.value):
            h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0:
                self.handles.append(h)

    def sample(self):
        gpus = []
        for i, h in enumerate(self.handles):
            g = {"idx": i}
            mw = ctypes.c_uint()
            if self.lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0:
                g["power_w"] = mw.value / 1000.0
            mj = ctypes.c_ulonglong()
            if self.lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(mj)) == 0:
                g["energy_j"] = mj.value / 1000.0
            u = _Utilization()
            if self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0:
                g["util_pct"] = u.gpu
            m = _Memory()
            if self.lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) == 0:
                g["mem_used_mb"] = m.used / 2**20
            t = ctypes.c_uint()
            if self.lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t)) == 0:   # 0 = NVML_TEMPERATURE_GPU
                g["temp_c"] = t.value
            gpus.append(g)
        return gpus

    def close(self):
        try:
            self.lib.nvmlShutdown()
        except Exception:
            pass


class NodeTelemetry:
    """Background sampler for the GPUs of this node. `stop()` writes and returns the energy summary."""

    def __init__(self, out_dir, every_s=30.0, tag=None):
        self.host = socket.gethostname()
        self.path = Path(out_dir) / "telemetry" / f"{self.host}.jsonl"
        self.every_s = max(1.0, float(every_s))
        self.tag = tag or {}
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._first = self._last = None          # (time, gpus) of the first and last sample
        self._power_j = {}                       # fallback: integrated power per GPU
        self._prev = None

    def start(self):
        try:
            self._nvml = NVML()
        except Exception as e:
            print(f"[telemetry] off on {self.host}: {e}", flush=True)
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-telemetry")
        self._thread.start()
        return self

    def _write(self, rec):
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _take(self):
        now = time.time()
        gpus = self._nvml.sample()
        if self._prev is not None:
            dt = now - self._prev[0]
            for g, p in zip(gpus, self._prev[1]):
                if "power_w" in g and "power_w" in p:
                    self._power_j[g["idx"]] = self._power_j.get(g["idx"], 0.0) + 0.5 * (g["power_w"] + p["power_w"]) * dt
        self._prev = (now, gpus)
        if self._first is None:
            self._first = (now, gpus)
        self._last = (now, gpus)
        self._write({"t": round(now, 1), "host": self.host, **self.tag, "gpus": gpus})

    def _run(self):
        while not self._stop.is_set():
            try:
                self._take()
            except Exception as e:
                print(f"[telemetry] sample failed on {self.host}: {e}", flush=True)
            self._stop.wait(self.every_s)

    def summary(self):
        if self._first is None or self._last is None:
            return None
        t0, g0 = self._first
        t1, g1 = self._last
        per_gpu = []
        for a, b in zip(g0, g1):
            if "energy_j" in a and "energy_j" in b:
                per_gpu.append(b["energy_j"] - a["energy_j"])
            else:
                per_gpu.append(self._power_j.get(a["idx"], 0.0))
        dur = t1 - t0
        return {"summary": True, "host": self.host, **self.tag, "t0": round(t0, 1), "t1": round(t1, 1),
                "duration_s": round(dur, 1), "n_gpus": len(per_gpu), "gpu_hours": round(len(per_gpu) * dur / 3600, 4),
                "energy_kwh": round(sum(per_gpu) / 3.6e6, 6),
                "energy_source": "nvml_counter" if all("energy_j" in g for g in g1) else "power_integral"}

    def stop(self):
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=10)
        try:
            self._take()                         # a last sample at the very end
        except Exception:
            pass
        s = self.summary()
        if s is not None:
            self._write(s)
        self._nvml.close()
        self._thread = None
        return s
