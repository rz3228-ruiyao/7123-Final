"""
Run logger for the ScienceQA LoRA experiments.

Append-only JSONL log of every training run. Each line records:
- timestamp, run_id, git_commit (if available)
- full CFG snapshot
- training history (loss + val acc per epoch, including per-num_choices)
- final TTA result
- public leaderboard score (added manually after submission)
- notes

Usage:
    from run_logger import RunLogger
    logger = RunLogger("runs/runs.jsonl")
    run_id = logger.start_run(cfg, notes="baseline + LoRA r=16")
    ...
    logger.log_epoch(run_id, epoch=1, loss=0.56, val_acc=0.7525,
                     by_num_choices={2: 0.81, 3: 0.72, 4: 0.89, 5: 0.16})
    ...
    logger.finalize(run_id, best_val_acc=0.7750,
                    tta_val_acc=0.8050, public_lb=0.8008)
    logger.show_table()
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "n/a"


def _serialize_cfg(cfg: Any) -> dict:
    """Turn a dataclass / dict / object into a JSON-friendly dict."""
    if is_dataclass(cfg):
        d = asdict(cfg)
    elif isinstance(cfg, dict):
        d = dict(cfg)
    else:
        d = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")
             and not callable(getattr(cfg, k))}
    out = {}
    for k, v in d.items():
        if isinstance(v, (Path,)):
            out[k] = str(v)
        elif isinstance(v, (tuple, list, set)):
            out[k] = list(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


class RunLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    # ── start / append / finalize ────────────────────────────────────────
    def start_run(self, cfg, notes: str = "") -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "notes": notes,
            "cfg": _serialize_cfg(cfg),
            "history": [],
            "best_val_acc": None,
            "tta_val_acc": None,
            "tta_by_num_choices": None,
            "public_lb": None,
            "private_lb": None,
            "finished_at": None,
            "status": "running",
        }
        self._append(record)
        return run_id

    def log_epoch(self, run_id: str, epoch: int, loss: float,
                  val_acc: float | None = None,
                  by_num_choices: dict | None = None,
                  lr: float | None = None) -> None:
        rec = self._find(run_id)
        rec.setdefault("history", []).append({
            "epoch": epoch, "loss": float(loss),
            "val_acc": None if val_acc is None else float(val_acc),
            "by_num_choices": by_num_choices or {},
            "lr": None if lr is None else float(lr),
            "logged_at": datetime.now().isoformat(timespec="seconds"),
        })
        self._update(rec)

    def finalize(self, run_id: str,
                 best_val_acc: float | None = None,
                 tta_val_acc: float | None = None,
                 tta_by_num_choices: dict | None = None,
                 public_lb: float | None = None,
                 private_lb: float | None = None,
                 status: str = "done",
                 extra: dict | None = None) -> None:
        rec = self._find(run_id)
        if best_val_acc is not None:
            rec["best_val_acc"] = float(best_val_acc)
        if tta_val_acc is not None:
            rec["tta_val_acc"] = float(tta_val_acc)
        if tta_by_num_choices is not None:
            rec["tta_by_num_choices"] = tta_by_num_choices
        if public_lb is not None:
            rec["public_lb"] = float(public_lb)
        if private_lb is not None:
            rec["private_lb"] = float(private_lb)
        if extra:
            rec.update(extra)
        rec["finished_at"] = datetime.now().isoformat(timespec="seconds")
        rec["status"] = status
        self._update(rec)

    def add_lb_score(self, run_id: str, public_lb: float | None = None,
                     private_lb: float | None = None) -> None:
        """Call this after Kaggle submission to attach the score."""
        rec = self._find(run_id)
        if public_lb is not None:
            rec["public_lb"] = float(public_lb)
        if private_lb is not None:
            rec["private_lb"] = float(private_lb)
        self._update(rec)

    # ── reading and viewing ──────────────────────────────────────────────
    def all_runs(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def show_table(self, fields: list[str] | None = None) -> None:
        runs = self.all_runs()
        if not runs:
            print("(no runs logged yet)")
            return
        rows = []
        for r in runs:
            cfg = r.get("cfg", {})
            rows.append({
                "run_id": r["run_id"],
                "epochs": cfg.get("epochs"),
                "lr": cfg.get("lr"),
                "lora_r": cfg.get("lora_r"),
                "img_size": cfg.get("img_size"),
                "permute": cfg.get("permute_choices"),
                "best_val": r.get("best_val_acc"),
                "tta_val": r.get("tta_val_acc"),
                "public_lb": r.get("public_lb"),
                "notes": (r.get("notes") or "")[:40],
            })
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
        except ImportError:
            for r in rows:
                print(r)

    # ── internals: write the whole file each time (small, simpler) ──────
    def _append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _find(self, run_id: str) -> dict:
        runs = self.all_runs()
        for r in runs:
            if r["run_id"] == run_id:
                return r
        raise KeyError(f"run_id {run_id} not found in {self.path}")

    def _update(self, record: dict) -> None:
        runs = self.all_runs()
        runs = [r for r in runs if r["run_id"] != record["run_id"]] + [record]
        runs.sort(key=lambda r: r["run_id"])
        with self.path.open("w") as f:
            for r in runs:
                f.write(json.dumps(r) + "\n")
