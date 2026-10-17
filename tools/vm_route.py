#!/usr/bin/env python3
"""Route Voicemeeter's virtual inputs to its B buses and verify every write.

Each pair then acts as a plain virtual cable (Milestone 1 / Stage 05-06
two-cable routes, Windows):

    play "Voicemeeter Input"        -> record "Voicemeeter Out B1"
    play "Voicemeeter AUX Input"    -> record "Voicemeeter Out B2"   (Banana, Potato)
    play "Voicemeeter VAIO3 Input"  -> record "Voicemeeter Out B3"   (Potato)

Every other bus send on those strips is cleared, so nothing reaches a speaker.
The virtual strips come after the hardware ones, so their index depends on the
edition: Voicemeeter Strip[2], Banana Strip[3..4], Potato Strip[5..7].

Remote API writes are never trusted (BUG-002, Documents/Bug Report): while
Voicemeeter runs elevated, a non-elevated client reads every parameter, but its
writes return 0 and are dropped. So this script

  * refuses to write when Voicemeeter's integrity level differs from its own,
  * reads every write back and fails when it does not land, and
  * never launches Voicemeeter from an elevated process (it would start
    elevated and lock out every later non-elevated client).

Usage:
    vm_route.py [--edition banana] [--timeout 2]   apply the routing, verify it
    vm_route.py --check          integrity levels + current routing, no writes
    vm_route.py --no-preflight   skip the integrity check (the read-back still
                                 catches a dropped write)
    vm_route.py --self-test

Exit status: 0 = routing applied and verified (--check: integrity levels
match), 1 = a write did not land, 2 = Voicemeeter not installed / not running /
integrity level mismatch. Windows only (the self-test runs anywhere).
Standard library only.
"""
import argparse
import os
import platform
import sys
import time

# VBVMR_GetVoicemeeterType -> (name, hardware strips, virtual input endpoints,
# per-strip bus sends). Virtual input k is routed to the k-th B bus.
EDITIONS = {
    1: ("Voicemeeter", 2, ("Voicemeeter Input",), ("A1", "B1")),
    2: ("Voicemeeter Banana", 3, ("Voicemeeter Input", "Voicemeeter AUX Input"),
        ("A1", "A2", "A3", "B1", "B2")),
    3: ("Voicemeeter Potato", 5,
        ("Voicemeeter Input", "Voicemeeter AUX Input", "Voicemeeter VAIO3 Input"),
        ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3")),
}
# VBVMR_RunVoicemeeter type per edition; the x64 builds are +3.
EDITION_CODES = {"standard": 1, "banana": 2, "potato": 3}
VOICEMEETER_EXES = {"voicemeeter.exe", "voicemeeter_x64.exe", "voicemeeterpro.exe",
                    "voicemeeterpro_x64.exe", "voicemeeter8.exe", "voicemeeter8x64.exe"}

UNINSTALL_KEY = (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
                 r"\VB:Voicemeeter {17359A74-1236-5467}")
DEFAULT_DIR = r"C:\Program Files (x86)\VB\Voicemeeter"

# Mandatory integrity level RIDs (winnt.h SECURITY_MANDATORY_*_RID).
MEDIUM, HIGH = 0x2000, 0x3000
IL_NAMES = {0x0000: "Untrusted", 0x1000: "Low", MEDIUM: "Medium", 0x2100: "Medium Plus",
            HIGH: "High", 0x4000: "System"}

TH32CS_SNAPPROCESS = 0x2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x8
TOKEN_INTEGRITY_LEVEL = 25  # TOKEN_INFORMATION_CLASS TokenIntegrityLevel


def il_name(rid):
    if rid is None:
        return "unreadable (elevated?)"
    return IL_NAMES.get(rid, hex(rid))


def preflight(own_il, procs):
    """procs: [(pid, exe, integrity RID or None)]. Returns an error message, or
    None when every running Voicemeeter shares our integrity level."""
    for pid, exe, il in procs:
        if own_il is None or il != own_il:
            return (f"{exe} (pid {pid}) runs at {il_name(il)} integrity, this script at "
                    f"{il_name(own_il)}.\nVoicemeeter drops Remote API writes across that "
                    "boundary while reporting success (BUG-002).\nExit Voicemeeter and start "
                    "it again normally (not 'Run as administrator'), or run this script "
                    "from a shell with the same elevation.")
    return None


def launch_code(edition, own_il, win64):
    """VBVMR_RunVoicemeeter type, or None when this process is elevated: the
    mixer would inherit the elevation (BUG-002)."""
    if own_il is None or own_il >= HIGH:
        return None
    return EDITION_CODES[edition] + (3 if win64 else 0)


def plan_routes(vm_type):
    """[(strip index, playback endpoint, bus)]: virtual input k -> k-th B bus."""
    _, hardware, inputs, sends = EDITIONS[vm_type]
    b_buses = [s for s in sends if s.startswith("B")]
    return [(hardware + k, endpoint, b_buses[k]) for k, endpoint in enumerate(inputs)]


def same(a, b):
    return abs(a - b) < 1e-6


def routing(api, vm_type):
    """{strip index: [buses it sends to]} for the virtual strips."""
    api.is_dirty()  # refreshes the client's copy of the parameters
    sends = EDITIONS[vm_type][3]
    return {strip: [bus for bus in sends if api.get(f"Strip[{strip}].{bus}")[1] >= 0.5]
            for strip, _, _ in plan_routes(vm_type)}


def set_verified(api, name, value, timeout, poll=0.02):
    """Set a parameter and wait until it reads back. Returns an error message
    or None. A 0 from the Set call only means the request was queued."""
    rc = api.set(name, value)
    if rc != 0:
        return f"{name} = {value:g}: VBVMR_SetParameterFloat returned {rc}"
    deadline = time.monotonic() + timeout
    while True:
        api.is_dirty()
        rc, got = api.get(name)
        if rc == 0 and same(got, value):
            return None
        if time.monotonic() >= deadline:
            return (f"{name} = {value:g} returned 0 but still reads {got:g} after "
                    f"{timeout:g} s: Voicemeeter dropped the write (BUG-002 - is it "
                    "running elevated?)")
        time.sleep(poll)


def apply(api, vm_type, timeout):
    """Route every virtual strip to its bus only. Writes just the sends that
    differ and stops at the first one that does not land.
    Returns (names written, error message or None)."""
    api.is_dirty()
    written = []
    for strip, _, target in plan_routes(vm_type):
        for bus in EDITIONS[vm_type][3]:
            name = f"Strip[{strip}].{bus}"
            want = 1.0 if bus == target else 0.0
            rc, got = api.get(name)
            if rc == 0 and same(got, want):
                continue
            err = set_verified(api, name, want, timeout)
            if err:
                return written, err
            written.append(name)
    return written, None


class Win32:
    """Process list and token integrity levels (kernel32/advapi32 via ctypes)."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes as w
        self.c, self.w = ctypes, w
        k32 = self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        adv = self.adv = ctypes.WinDLL("advapi32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.argtypes = (w.DWORD, w.DWORD)
        k32.CreateToolhelp32Snapshot.restype = w.HANDLE
        k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = (w.HANDLE, ctypes.c_void_p)
        k32.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
        k32.OpenProcess.restype = w.HANDLE
        k32.GetCurrentProcess.restype = w.HANDLE
        k32.CloseHandle.argtypes = (w.HANDLE,)
        adv.OpenProcessToken.argtypes = (w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE))
        adv.GetTokenInformation.argtypes = (w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD,
                                            ctypes.POINTER(w.DWORD))
        adv.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
        adv.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        adv.GetSidSubAuthority.argtypes = (ctypes.c_void_p, w.DWORD)
        adv.GetSidSubAuthority.restype = ctypes.POINTER(w.DWORD)

    def _token_integrity(self, process):
        c, w = self.c, self.w
        token = w.HANDLE()
        if not self.adv.OpenProcessToken(process, TOKEN_QUERY, c.byref(token)):
            return None
        try:
            size = w.DWORD()
            self.adv.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, None, 0, c.byref(size))
            buf = c.create_string_buffer(size.value)
            if not self.adv.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, buf, size,
                                                c.byref(size)):
                return None
            sid = c.cast(buf, c.POINTER(c.c_void_p))[0]  # TOKEN_MANDATORY_LABEL.Label.Sid
            n = self.adv.GetSidSubAuthorityCount(sid)[0]
            return self.adv.GetSidSubAuthority(sid, n - 1)[0]
        finally:
            self.k32.CloseHandle(token)

    def own_integrity(self):
        return self._token_integrity(self.k32.GetCurrentProcess())

    def integrity(self, pid):
        """Integrity RID of a process, or None when its token cannot be read
        (from a non-elevated caller that means it runs elevated)."""
        h = self.k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            return self._token_integrity(h)
        finally:
            self.k32.CloseHandle(h)

    def voicemeeter_processes(self):
        """[(pid, exe, integrity RID or None)] of running Voicemeeter mixers."""
        c, w = self.c, self.w

        class Entry(c.Structure):  # PROCESSENTRY32W
            _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("th32ProcessID", w.DWORD),
                        ("th32DefaultHeapID", c.c_size_t), ("th32ModuleID", w.DWORD),
                        ("cntThreads", w.DWORD), ("th32ParentProcessID", w.DWORD),
                        ("pcPriClassBase", w.LONG), ("dwFlags", w.DWORD),
                        ("szExeFile", w.WCHAR * 260)]

        snap = self.k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap in (None, c.c_void_p(-1).value):
            raise c.WinError(c.get_last_error())
        found = []
        try:
            e = Entry()
            e.dwSize = c.sizeof(Entry)
            ok = self.k32.Process32FirstW(snap, c.byref(e))
            while ok:
                if e.szExeFile.lower() in VOICEMEETER_EXES:
                    found.append((e.th32ProcessID, e.szExeFile))
                ok = self.k32.Process32NextW(snap, c.byref(e))
        finally:
            self.k32.CloseHandle(snap)
        return [(pid, exe, self.integrity(pid)) for pid, exe in found]


class Remote:
    """The VoicemeeterRemote DLL calls used here; return codes pass through."""

    def __init__(self, path):
        import ctypes
        self.c = ctypes
        d = self.dll = ctypes.WinDLL(path)
        d.VBVMR_SetParameterFloat.argtypes = (ctypes.c_char_p, ctypes.c_float)
        d.VBVMR_GetParameterFloat.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_float))
        d.VBVMR_GetVoicemeeterType.argtypes = (ctypes.POINTER(ctypes.c_long),)
        d.VBVMR_GetVoicemeeterVersion.argtypes = (ctypes.POINTER(ctypes.c_long),)
        d.VBVMR_RunVoicemeeter.argtypes = (ctypes.c_long,)

    def login(self):
        return self.dll.VBVMR_Login()

    def logout(self):
        return self.dll.VBVMR_Logout()

    def run(self, code):
        return self.dll.VBVMR_RunVoicemeeter(code)

    def is_dirty(self):
        return self.dll.VBVMR_IsParametersDirty()

    def _long(self, fn):
        v = self.c.c_long()
        return fn(self.c.byref(v)), v.value

    def vm_type(self):
        return self._long(self.dll.VBVMR_GetVoicemeeterType)

    def version(self):
        return self._long(self.dll.VBVMR_GetVoicemeeterVersion)

    def get(self, name):
        v = self.c.c_float()
        return self.dll.VBVMR_GetParameterFloat(name.encode(), self.c.byref(v)), v.value

    def set(self, name, value):
        return self.dll.VBVMR_SetParameterFloat(name.encode(), value)


def find_dll():
    """VoicemeeterRemote DLL matching this Python's bitness, or None."""
    import ctypes
    import winreg
    dirs = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, UNINSTALL_KEY, 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_32KEY) as k:
            dirs.append(os.path.dirname(winreg.QueryValueEx(k, "UninstallString")[0].strip('"')))
    except OSError:
        pass
    dirs.append(DEFAULT_DIR)
    name = "VoicemeeterRemote64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8 else "VoicemeeterRemote.dll"
    for d in dirs:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    return None


def wait_ready(api, seconds=10.0):
    """After VBVMR_RunVoicemeeter. A session that logged in before the mixer
    started never connects to it, so log in again until it answers."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.5)
        api.logout()
        if api.login() == 0 and api.vm_type()[0] == 0 and api.is_dirty() >= 0:
            return True
    return False


def describe(api, vm_type):
    now = routing(api, vm_type)
    for strip, endpoint, target in plan_routes(vm_type):
        buses = " ".join(now[strip]) or "-"
        mark = "" if now[strip] == [target] else f"   (want {target} only)"
        print(f"  Strip[{strip}] {endpoint:<24} -> {buses}{mark}")


def run(args):
    dll = find_dll()
    if dll is None:
        print("vm_route: Voicemeeter is not installed (no VoicemeeterRemote DLL)", file=sys.stderr)
        return 2
    win = Win32()
    own = win.own_integrity()
    procs = win.voicemeeter_processes()
    print(f"this script: {il_name(own)} integrity")
    api = Remote(dll)
    rc = api.login()
    if rc < 0:
        print(f"vm_route: VBVMR_Login returned {rc}", file=sys.stderr)
        return 2
    try:
        if not procs:
            if args.check:
                print("Voicemeeter is not running")
                return 2
            win64 = (os.environ.get("PROCESSOR_ARCHITEW6432") or platform.machine()).endswith("64")
            code = launch_code(args.edition, own, win64)
            if code is None:
                print("vm_route: Voicemeeter is not running and this script is elevated; "
                      "launching it from here would start it elevated (BUG-002). Start "
                      "Voicemeeter normally, then rerun.", file=sys.stderr)
                return 2
            print(f"starting Voicemeeter ({args.edition}) ...")
            api.run(code)
            if not wait_ready(api):
                print("vm_route: Voicemeeter did not come up within 10 s", file=sys.stderr)
                return 2
            procs = win.voicemeeter_processes()
        for pid, exe, il in procs:
            print(f"{exe} (pid {pid}): {il_name(il)} integrity")
        mismatch = preflight(own, procs)
        if mismatch:
            print(f"vm_route: {mismatch}", file=sys.stderr)
            if not (args.check or args.no_preflight):
                return 2

        rc, vm_type = api.vm_type()
        if rc != 0 or vm_type not in EDITIONS:
            print(f"vm_route: unknown Voicemeeter type {vm_type} (rc {rc})", file=sys.stderr)
            return 2
        _, v = api.version()
        print(f"{EDITIONS[vm_type][0]} {v >> 24}.{v >> 16 & 255}.{v >> 8 & 255}.{v & 255}")
        print("routing now:")
        describe(api, vm_type)
        if args.check:
            return 2 if mismatch else 0

        written, err = apply(api, vm_type, args.timeout)
        for name in written:
            print(f"  set {name}")
        if err:
            print(f"vm_route: {err}", file=sys.stderr)
            return 1
        print("routing applied and verified:" if written else "routing already correct:")
        describe(api, vm_type)
        return 0
    finally:
        api.logout()


class FakeRemote:
    """In-memory Remote API. drop=True acts like an elevated Voicemeeter seen
    from a non-elevated client (BUG-002): Set returns 0, nothing changes."""

    def __init__(self, params=None, drop=False):
        self.params = dict(params or {})
        self.drop = drop
        self.writes = []

    def is_dirty(self):
        return 0

    def get(self, name):
        return 0, self.params.get(name, 0.0)

    def set(self, name, value):
        self.writes.append(name)
        if not self.drop:
            self.params[name] = value
        return 0


def self_test():
    # Banana's defaults send both virtual strips to A1 + B1.
    defaults = {f"Strip[{s}].{b}": 1.0 for s in (3, 4) for b in ("A1", "B1")}
    healthy = FakeRemote(defaults)
    h_written, h_err = apply(healthy, 2, timeout=0.0)
    settled = FakeRemote(healthy.params)
    s_written, s_err = apply(settled, 2, timeout=0.0)
    dropped = FakeRemote(defaults, drop=True)
    d_written, d_err = apply(dropped, 2, timeout=0.0)
    vm = "voicemeeterpro_x64.exe"

    cases = [
        # The virtual strips follow the hardware ones (the scratch script used Strip[2]).
        ("Banana strip layout", plan_routes(2) == [(3, "Voicemeeter Input", "B1"),
                                                   (4, "Voicemeeter AUX Input", "B2")]),
        ("Potato strip layout", [(s, b) for s, _, b in plan_routes(3)]
         == [(5, "B1"), (6, "B2"), (7, "B3")]),
        ("Voicemeeter strip layout", plan_routes(1) == [(2, "Voicemeeter Input", "B1")]),
        ("healthy: routed + verified", h_err is None
         and routing(healthy, 2) == {3: ["B1"], 4: ["B2"]}),
        ("healthy: only changed sends", h_written == ["Strip[3].A1", "Strip[4].A1",
                                                      "Strip[4].B1", "Strip[4].B2"]),
        ("already routed: no writes", s_err is None and s_written == [] and settled.writes == []),
        ("dropped write detected", d_err is not None and "BUG-002" in d_err),
        ("stops at first dropped write", d_written == [] and dropped.writes == ["Strip[3].A1"]),
        ("same integrity passes", preflight(MEDIUM, [(1, vm, MEDIUM)]) is None
         and preflight(HIGH, [(1, vm, HIGH)]) is None),
        ("not running passes", preflight(MEDIUM, []) is None),
        ("elevated Voicemeeter refused", "BUG-002" in (preflight(MEDIUM, [(1, vm, HIGH)]) or "")),
        ("unreadable token refused", preflight(MEDIUM, [(1, vm, None)]) is not None),
        ("lower Voicemeeter refused", preflight(HIGH, [(1, vm, MEDIUM)]) is not None),
        ("launch codes (x64 +3)", launch_code("banana", MEDIUM, True) == 5
         and launch_code("banana", MEDIUM, False) == 2 and launch_code("potato", MEDIUM, True) == 6),
        ("elevated script never launches", launch_code("banana", HIGH, True) is None
         and launch_code("banana", None, True) is None),
    ]
    ok = True
    for name, good in cases:
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true",
                   help="print integrity levels and the current routing; write nothing")
    p.add_argument("--no-preflight", action="store_true",
                   help="write even when the integrity levels differ (the read-back still "
                        "catches a dropped write)")
    p.add_argument("--edition", choices=sorted(EDITION_CODES), default="banana",
                   help="edition to start when Voicemeeter is not running (default banana)")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="seconds to wait for each write to read back (default 2)")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if sys.platform != "win32":
        print("vm_route: Voicemeeter is Windows-only", file=sys.stderr)
        return 2
    sys.stdout.reconfigure(line_buffering=True)  # keep stdout/stderr in order when piped
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
