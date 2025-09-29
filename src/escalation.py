# S1: Escalation counters + policy
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Escalator:
    th_suggest: float = 0.50
    th_warn: float = 0.70
    th_enforce: float = 0.90
    memory: Dict[str, int] = field(default_factory=dict)  # per-key repeat count

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
