from dataclasses import dataclass, field
from typing import Any, Callable

from .view_sampler import ViewSamplerCfg


@dataclass(kw_only=True)
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
    target_poses: list[Any] = field(default_factory=list)
    context_indices: list[int] = field(default_factory=list)
    target_names: list[Any] = field(default_factory=list)
    _observers: dict[str, list[Callable[[Any], None]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def register_observer(self, field_name: str, observer: Callable[[Any], None]) -> None:
        """Register a callback that fires whenever the specified field is reassigned."""
        if field_name not in self._observers:
            self._observers[field_name] = []
        self._observers[field_name].append(observer)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        observers = getattr(self, "_observers", None)
        if observers and name in observers:
            for observer in observers[name]:
                observer(value)
