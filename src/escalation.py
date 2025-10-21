# escalation.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml, os

def _load_cfg(path: str = "config.yaml") -> dict:
    # minimal loader; assumes file at project root
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

@dataclass
class Escalator:
    th_suggest: float = 0.50
    th_warn: float = 0.70
    th_enforce: float = 0.90
    memory: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "Escalator":
        cfg = _load_cfg(path)
        esc = (cfg.get("escalation") or {})
        return cls(
            th_suggest=float(esc.get("suggest", cls.th_suggest)),
            th_warn=float(esc.get("warn", cls.th_warn)),
            th_enforce=float(esc.get("enforce", cls.th_enforce)),
        )

    def base_action(self, p: float) -> str:
        if p >= self.th_enforce: return "enforce"
        if p >= self.th_warn:    return "warn"
        if p >= self.th_suggest: return "suggest"
        return "pass"

    def step(self, key: str, prob: float) -> str:
        base = self.base_action(prob)
        if base == "pass":
            self.memory.pop(key, None)
            return "pass"
        c = self.memory.get(key, 0)
        ladder = ["suggest", "warn", "enforce"]
        idx = min(c, len(ladder)-1)
        act = ladder[max(ladder.index(base), idx)]
        self.memory[key] = c + 1
        return act
