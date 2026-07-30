"""Small cross-process locks for long-running experiment schedulers.

The lock file is created atomically and records the owning process.  Locks
left behind by a dead process on the same host are reclaimed automatically.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class ActiveRunError(RuntimeError):
    """Raised when another live process already owns an experiment lock."""


@dataclass(frozen=True)
class RunLock:
    path: Path
    token: str


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows ``os.kill(pid, 0)`` terminates the target process instead
        # of providing the POSIX existence probe.  Query the process handle and
        # exit code through the read-only Win32 API instead.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            # Access denied still means the process exists; an invalid PID does
            # not return a handle.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return error.errno == errno.EPERM
    return True


def _read_lock(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _existing_lock_is_active(path: Path) -> tuple[bool, dict[str, object] | None]:
    payload = _read_lock(path)
    if payload is None:
        # A competing process may have created the file but not finished writing
        # it yet.  Only reclaim a malformed lock after a conservative grace
        # period, avoiding a race that would admit two owners.
        try:
            age_seconds = max(0.0, time.time() - path.stat().st_mtime)
        except FileNotFoundError:
            return False, None
        return age_seconds < 600.0, None

    owner_host = str(payload.get("hostname", ""))
    try:
        owner_pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        owner_pid = -1

    if owner_host and owner_host != socket.gethostname():
        # A PID cannot be checked safely on another host sharing the same
        # results volume, so preserve that lock rather than risking corruption.
        return True, payload
    return _pid_is_alive(owner_pid), payload


def acquire_run_lock(path: str | Path, *, label: str) -> RunLock:
    """Atomically acquire ``path`` or raise ``ActiveRunError``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": time.time(),
        "label": label,
    }

    for _ in range(3):
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            active, owner = _existing_lock_is_active(path)
            if active:
                owner_text = (
                    f"pid={owner.get('pid')} host={owner.get('hostname')}"
                    if owner
                    else "owner metadata is still being written"
                )
                raise ActiveRunError(f"{label} is already running ({owner_text})")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        return RunLock(path=path, token=token)

    raise ActiveRunError(f"could not acquire {label} lock: {path}")


def release_run_lock(lock: RunLock | None) -> None:
    """Release a lock only when it is still owned by ``lock.token``."""

    if lock is None:
        return
    payload = _read_lock(lock.path)
    if payload is None or payload.get("token") != lock.token:
        return
    try:
        lock.path.unlink()
    except FileNotFoundError:
        pass
