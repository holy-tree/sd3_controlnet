"""Named-parameter EMA with CPU storage and temporary weight swapping."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch


class ModelEMA:
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_interval: int = 1,
        use_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        parameters = list(named_parameters)
        if not parameters:
            raise ValueError("EMA requires at least one parameter")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be in [0, 1]")
        if update_after_step < 0:
            raise ValueError("EMA update_after_step must be non-negative")
        if update_interval <= 0:
            raise ValueError("EMA update_interval must be positive")
        if inv_gamma <= 0.0 or power <= 0.0:
            raise ValueError("EMA inv_gamma and power must be positive")
        names = [name for name, _ in parameters]
        if len(names) != len(set(names)):
            raise ValueError("EMA parameter names must be unique")

        self.parameter_names = tuple(names)
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_interval = int(update_interval)
        self.use_warmup = bool(use_warmup)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.device = torch.device(device)
        self.dtype = dtype
        self.optimization_step = 0
        self.num_updates = 0
        self.cur_decay_value = 0.0
        self.shadow_params = [
            parameter.detach().to(device=self.device, dtype=self.dtype).clone()
            for _, parameter in parameters
        ]

    def _validate_parameters(
        self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]
    ) -> list[torch.nn.Parameter]:
        parameters = list(named_parameters)
        names = tuple(name for name, _ in parameters)
        if names != self.parameter_names:
            raise ValueError("EMA parameter names or order changed")
        values = [parameter for _, parameter in parameters]
        for name, shadow, parameter in zip(self.parameter_names, self.shadow_params, values):
            if shadow.shape != parameter.shape:
                raise ValueError(
                    f"EMA parameter shape mismatch for {name}: "
                    f"shadow={tuple(shadow.shape)}, parameter={tuple(parameter.shape)}"
                )
        return values

    def _current_decay(self) -> float:
        if not self.use_warmup:
            return self.decay
        warmup_decay = 1.0 - (1.0 + self.num_updates / self.inv_gamma) ** -self.power
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def step(
        self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]
    ) -> bool:
        parameters = self._validate_parameters(named_parameters)
        self.optimization_step += 1
        if self.optimization_step <= self.update_after_step:
            return False
        if (self.optimization_step - self.update_after_step) % self.update_interval != 0:
            return False

        self.num_updates += 1
        decay = self._current_decay()
        self.cur_decay_value = decay
        one_minus_decay = 1.0 - decay
        for shadow, parameter in zip(self.shadow_params, parameters):
            current = parameter.detach().to(device=self.device, dtype=self.dtype)
            shadow.mul_(decay).add_(current, alpha=one_minus_decay)
        return True

    @torch.no_grad()
    def _swap(
        self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]
    ) -> None:
        parameters = self._validate_parameters(named_parameters)
        for shadow, parameter in zip(self.shadow_params, parameters):
            current = parameter.detach().to(device=self.device, dtype=self.dtype).clone()
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
            shadow.copy_(current)

    @contextmanager
    def average_parameters(
        self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]
    ):
        parameters = list(named_parameters)
        self._swap(parameters)
        try:
            yield
        finally:
            self._swap(parameters)

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "parameter_names": list(self.parameter_names),
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_interval": self.update_interval,
            "use_warmup": self.use_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "optimization_step": self.optimization_step,
            "num_updates": self.num_updates,
            "cur_decay_value": self.cur_decay_value,
            "dtype": str(self.dtype).removeprefix("torch."),
            "shadow_params": [parameter.detach().cpu() for parameter in self.shadow_params],
        }

    def load_state_dict(self, state: dict) -> None:
        names = tuple(state.get("parameter_names", ()))
        if names != self.parameter_names:
            raise ValueError("EMA checkpoint parameter names do not match the current model")
        saved = state.get("shadow_params")
        if not isinstance(saved, list) or len(saved) != len(self.shadow_params):
            raise ValueError("EMA checkpoint has an invalid shadow parameter list")
        for name, target, source in zip(self.parameter_names, self.shadow_params, saved):
            if target.shape != source.shape:
                raise ValueError(
                    f"EMA checkpoint shape mismatch for {name}: "
                    f"saved={tuple(source.shape)}, expected={tuple(target.shape)}"
                )
            target.copy_(source.to(device=self.device, dtype=self.dtype))

        for key in ("decay", "inv_gamma", "power"):
            saved_value = float(state.get(key, getattr(self, key)))
            if saved_value != getattr(self, key):
                raise ValueError(
                    f"EMA checkpoint config mismatch for {key}: "
                    f"saved={saved_value}, current={getattr(self, key)}"
                )
        for key in ("update_after_step", "update_interval"):
            saved_value = int(state.get(key, getattr(self, key)))
            if saved_value != getattr(self, key):
                raise ValueError(
                    f"EMA checkpoint config mismatch for {key}: "
                    f"saved={saved_value}, current={getattr(self, key)}"
                )
        saved_warmup = bool(state.get("use_warmup", self.use_warmup))
        if saved_warmup != self.use_warmup:
            raise ValueError(
                "EMA checkpoint config mismatch for use_warmup: "
                f"saved={saved_warmup}, current={self.use_warmup}"
            )
        self.optimization_step = int(state.get("optimization_step", 0))
        self.num_updates = int(state.get("num_updates", 0))
        self.cur_decay_value = float(state.get("cur_decay_value", 0.0))

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_state_dict(state)
