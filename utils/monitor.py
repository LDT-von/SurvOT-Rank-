#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@file: monitor.py
@desc: 后台进程监控器（CPU / RAM / GPU / 磁盘 / 训练 meta）。
       设计目标:
         1. 零阻塞: 在 daemon 线程里按固定间隔采样，训练主循环不感知。
         2. 零依赖: 只用 psutil (CPU/RAM/磁盘/网络) + torch (GPU)。
            psutil 未装时降级到只写 torch 能拿到的 GPU 信息 + 时间戳，绝不崩溃。
         3. 统一接口: 通过 args.results_dir/<csv> 落盘，按 fold 自动切文件。
         4. 与 train.py 解耦: 调用方只需 4 行代码。

用法（嵌入 train.py）:
    from utils.monitor import ProcessMonitor

    monitor = ProcessMonitor(
        results_dir=args.results_dir,
        fold=fold,
        interval_sec=2.0,
        enable_gpu=torch.cuda.is_available(),
        log_file=log_file,             # 可选: 把 INFO 也写进同一份 log
    )
    monitor.start()

    for epoch in range(args.max_epochs):
        monitor.set_meta(phase="train", epoch=epoch, fold=fold)
        for batch_idx, data in enumerate(loader):
            monitor.set_meta(batch=batch_idx, last_loss=loss.item())
            ...

    monitor.stop()
"""

from __future__ import annotations

import csv
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from typing import Any, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _safe_write_line(log_file, message):
    try:
        if log_file is not None:
            log_file.write(message + "\n")
            log_file.flush()
    except OSError:
        pass


def _try_import_torch():
    try:
        import torch  # noqa
        return torch
    except ImportError:
        return None


class ProcessMonitor:
    """
    后台线程采样器。每隔 interval_sec 秒采集一次系统 + 训练状态，
    append 一行到 csv_path。调用 set_meta() 注入训练上下文（epoch/batch/loss 等），
    这些字段会随下一帧采样一起写出。

    列定义（CSV header）:
        timestamp, iso_time, phase, fold, epoch, batch, last_loss,
        cpu_percent, cpu_count, ram_used_gb, ram_total_gb, ram_percent,
        gpu_count, gpu_name, gpu_util_percent, gpu_mem_used_gb, gpu_mem_total_gb,
        gpu_power_w, gpu_temp_c,
        disk_used_gb, disk_free_gb, disk_percent,
        net_sent_mb, net_recv_mb, net_sent_rate_mbps, net_recv_rate_mbps,
        pid, rss_gb, threads, fds, children
    """

    HEADER = [
        "timestamp", "iso_time", "phase", "fold", "epoch", "batch", "last_loss",
        "cpu_percent", "cpu_count", "ram_used_gb", "ram_total_gb", "ram_percent",
        "gpu_count", "gpu_name", "gpu_util_percent", "gpu_mem_used_gb", "gpu_mem_total_gb",
        "gpu_power_w", "gpu_temp_c",
        "disk_used_gb", "disk_free_gb", "disk_percent",
        "net_sent_mb", "net_recv_mb", "net_sent_rate_mbps", "net_recv_rate_mbps",
        "pid", "rss_gb", "threads", "fds", "children",
    ]

    def __init__(
        self,
        results_dir: str,
        fold: Optional[int] = None,
        interval_sec: float = 2.0,
        enable_gpu: bool = True,
        log_file=None,
        csv_name: Optional[str] = None,
    ):
        self.results_dir = results_dir
        self.fold = fold
        self.interval = max(0.5, float(interval_sec))
        self.enable_gpu = bool(enable_gpu)
        self.log_file = log_file
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._meta_lock = threading.Lock()
        self._meta: dict[str, Any] = {
            "phase": "init", "epoch": -1, "batch": -1, "last_loss": None,
        }
        self._torch = _try_import_torch() if enable_gpu else None
        self._process = None
        self._gpu_handle = None
        self._gpu_name = ""
        self._last_net: Optional[tuple[float, float, float]] = None  # (sent, recv, t)

        # csv path
        os.makedirs(results_dir, exist_ok=True)
        if csv_name is None:
            suffix = f"_fold{fold}" if fold is not None else ""
            csv_name = f"process_monitor{suffix}.csv"
        self.csv_path = os.path.join(results_dir, csv_name)

        # try to get a GPU handle for power/temperature (nvidia-smi backed)
        if self._torch is not None and self.enable_gpu:
            try:
                if self._torch.cuda.is_available():
                    self._gpu_handle = self._torch.cuda.current_device()
                    self._gpu_name = self._torch.cuda.get_device_name(self._gpu_handle)
            except Exception:
                self._gpu_handle = None

        if _HAS_PSUTIL:
            try:
                self._process = psutil.Process(os.getpid())
                # warm up cpu_percent so first real sample is accurate
                psutil.cpu_percent(interval=None)
            except Exception:
                self._process = None

    # ---------------- public API ----------------

    def set_meta(self, **kwargs):
        """注入训练上下文；会被下一帧采样一起写出。线程安全。"""
        with self._meta_lock:
            self._meta.update(kwargs)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ensure_header()
        self._thread = threading.Thread(
            target=self._run, name=f"ProcessMonitor-{os.getpid()}", daemon=True
        )
        self._thread.start()
        msg = f"[monitor] started -> {self.csv_path} (interval={self.interval}s, gpu={self.enable_gpu})"
        print(msg)
        _safe_write_line(self.log_file, msg)

    def stop(self, timeout_sec: float = 5.0):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout_sec)
        if self._thread.is_alive():
            msg = f"[monitor] WARN: thread did not stop within {timeout_sec}s"
            print(msg)
            _safe_write_line(self.log_file, msg)
        else:
            msg = "[monitor] stopped"
            print(msg)
            _safe_write_line(self.log_file, msg)
        self._thread = None

    # ---------------- internals ----------------

    def _ensure_header(self):
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.HEADER)
            except OSError as err:
                msg = f"[monitor] WARN: cannot create {self.csv_path}: {err}"
                print(msg)
                _safe_write_line(self.log_file, msg)

    def _run(self):
        try:
            while not self._stop_event.is_set():
                row = self._sample()
                self._append(row)
                # sleep with early-exit on stop
                if self._stop_event.wait(self.interval):
                    break
        except Exception as err:
            msg = f"[monitor] FATAL: sampler crashed: {err}\n{traceback.format_exc()}"
            print(msg)
            _safe_write_line(self.log_file, msg)

    def _append(self, row: list):
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except OSError as err:
            # 磁盘满/被锁时只警告，不影响训练
            if getattr(err, "errno", None) == 28:
                msg = f"[monitor] WARN: disk full, skip row: {err}"
                print(msg)
                _safe_write_line(self.log_file, msg)
            else:
                msg = f"[monitor] WARN: append failed: {err}"
                print(msg)
                _safe_write_line(self.log_file, msg)

    def _sample(self) -> list:
        ts = time.time()
        iso = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        with self._meta_lock:
            meta = dict(self._meta)

        # ---- CPU / RAM ----
        cpu_percent = ""
        cpu_count = ""
        ram_used_gb = ""
        ram_total_gb = ""
        ram_percent = ""

        if _HAS_PSUTIL:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                cpu_percent = ""
            try:
                cpu_count = psutil.cpu_count(logical=True) or ""
            except Exception:
                cpu_count = ""
            try:
                vm = psutil.virtual_memory()
                ram_used_gb = round(vm.used / (1024 ** 3), 3)
                ram_total_gb = round(vm.total / (1024 ** 3), 3)
                ram_percent = vm.percent
            except Exception:
                pass

        # ---- GPU ----
        gpu_count = 0
        gpu_name = self._gpu_name
        gpu_util = ""
        gpu_mem_used = ""
        gpu_mem_total = ""
        gpu_power = ""
        gpu_temp = ""
        if self._torch is not None and self.enable_gpu:
            try:
                if self._torch.cuda.is_available():
                    gpu_count = self._torch.cuda.device_count()
                    if self._gpu_handle is None:
                        self._gpu_handle = self._torch.cuda.current_device()
                        gpu_name = self._torch.cuda.get_device_name(self._gpu_handle)
                    free, total = self._torch.cuda.mem_get_info(self._gpu_handle)
                    gpu_mem_used = round((total - free) / (1024 ** 3), 3)
                    gpu_mem_total = round(total / (1024 ** 3), 3)
            except Exception:
                pass
            # nvidia-smi via pynvml if available
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_handle or 0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                gpu_util = util.gpu
                try:
                    gpu_power = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)  # W
                except Exception:
                    gpu_power = ""
                try:
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    gpu_temp = ""
                pynvml.nvmlShutdown()
            except Exception:
                # pynvml 没装或失败，gpu_util / power / temp 留空
                pass

        # ---- Disk (results_dir 所在盘) ----
        disk_used = ""
        disk_free = ""
        disk_percent = ""
        try:
            du = psutil.disk_usage(self.results_dir)
            disk_used = round(du.used / (1024 ** 3), 3)
            disk_free = round(du.free / (1024 ** 3), 3)
            disk_percent = du.percent
        except Exception:
            pass

        # ---- Network ----
        net_sent = ""
        net_recv = ""
        net_sent_rate = ""
        net_recv_rate = ""
        if _HAS_PSUTIL:
            try:
                nc = psutil.net_io_counters()
                cur_sent, cur_recv = nc.bytes_sent, nc.bytes_recv
                if self._last_net is not None:
                    prev_sent, prev_recv, prev_t = self._last_net
                    dt = max(ts - prev_t, 1e-6)
                    net_sent_rate = round((cur_sent - prev_sent) * 8 / 1e6 / dt, 3)  # Mbps
                    net_recv_rate = round((cur_recv - prev_recv) * 8 / 1e6 / dt, 3)
                self._last_net = (cur_sent, cur_recv, ts)
                net_sent = round(cur_sent / (1024 ** 2), 3)
                net_recv = round(cur_recv / (1024 ** 2), 3)
            except Exception:
                pass

        # ---- Process ----
        pid = os.getpid()
        rss_gb = ""
        threads = ""
        fds = ""
        children = ""
        if self._process is not None:
            try:
                with self._process.oneshot():
                    rss_gb = round(self._process.memory_info().rss / (1024 ** 3), 3)
                    threads = self._process.num_threads()
                    try:
                        fds = self._process.num_fds()
                    except Exception:
                        fds = ""  # Windows: not available
                    try:
                        children = len(self._process.children(recursive=True))
                    except Exception:
                        children = ""
            except Exception:
                pass

        return [
            round(ts, 3),
            iso,
            meta.get("phase", ""),
            self.fold if self.fold is not None else "",
            meta.get("epoch", ""),
            meta.get("batch", ""),
            meta.get("last_loss", ""),
            cpu_percent, cpu_count, ram_used_gb, ram_total_gb, ram_percent,
            gpu_count, gpu_name, gpu_util, gpu_mem_used, gpu_mem_total,
            gpu_power, gpu_temp,
            disk_used, disk_free, disk_percent,
            net_sent, net_recv, net_sent_rate, net_recv_rate,
            pid, rss_gb, threads, fds, children,
        ]

    # ---------------- context manager ----------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ---------------- standalone smoke test ----------------

def _smoke_test():
    """直接 `python utils/monitor.py` 跑 5 秒验证能落盘。"""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_monitor_smoke")
    print(f"[smoke] writing to {out_dir}")
    m = ProcessMonitor(
        results_dir=out_dir,
        fold=0,
        interval_sec=1.0,
        enable_gpu=True,
    )
    m.start()
    for i in range(5):
        m.set_meta(phase="smoke", epoch=0, batch=i, last_loss=1.0 / (i + 1))
        time.sleep(1.0)
    m.stop()
    out_csv = os.path.join(out_dir, "process_monitor_fold0.csv")
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("[smoke] FAIL: csv not produced", file=sys.stderr)


if __name__ == "__main__":
    _smoke_test()