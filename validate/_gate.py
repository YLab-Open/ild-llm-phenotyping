"""Tiny pass/fail harness shared by the validators. Every gate prints a
one-page summary the author can read in a terminal before releasing GPUs."""
from __future__ import annotations
import sys


class Gate:
    def __init__(self, title: str):
        self.title = title
        self.checks = []          # (ok, name, detail)
        self.notes = []

    def check(self, ok: bool, name: str, detail: str = ""):
        self.checks.append((bool(ok), name, detail))
        return bool(ok)

    def note(self, s: str):
        self.notes.append(s)

    @property
    def failed(self):
        return [c for c in self.checks if not c[0]]

    def report(self, exit_on_fail: bool = True) -> bool:
        w = 78
        print("=" * w)
        print(self.title)
        print("=" * w)
        for ok, name, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}")
            if detail:
                for line in str(detail).splitlines():
                    print(f"         {line}")
        if self.notes:
            print("-" * w)
            print("  Notes (not gating):")
            for n in self.notes:
                for line in str(n).splitlines():
                    print(f"    {line}")
        print("=" * w)
        n_fail = len(self.failed)
        if n_fail:
            print(f"  RESULT: FAIL — {n_fail} of {len(self.checks)} checks failed")
        else:
            print(f"  RESULT: PASS — {len(self.checks)}/{len(self.checks)} checks")
        print("=" * w)
        if n_fail and exit_on_fail:
            sys.exit(1)
        return n_fail == 0
