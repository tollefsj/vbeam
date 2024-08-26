from typing import Callable, Literal, Optional, Tuple, Union, overload

from vbeam.fastmath import Array
from vbeam.fastmath import numpy as np
from vbeam.scan.base import CoordinateSystem, Scan
from vbeam.scan.util import parse_axes
from vbeam.util.arrays import grid


class LinearScan(Scan):
    x: Array
    y: Optional[Array]
    z: Array

    def get_points(self, flatten: bool = True) -> Array:
        shape = (self.num_points, 3) if flatten else (*self.shape, 3)
        y = np.array([0.0]) if self.y is None else self.y  # If the scan is 2D
        points = grid(self.x, y, self.z, shape=shape)
        return points

    def replace(
        self,
        x: Union[Array, Literal["unchanged"]] = "unchanged",
        y: Union[Array, Literal["unchanged"]] = "unchanged",
        z: Union[Array, Literal["unchanged"]] = "unchanged",
    ) -> "LinearScan":
        return LinearScan(
            x=x if x != "unchanged" else self.x,
            y=y if y != "unchanged" else self.y,
            z=z if z != "unchanged" else self.z,
        )

    def update(
        self,
        x: Optional[Callable[[Array], Array]] = None,
        y: Optional[Callable[[Array], Array]] = None,
        z: Optional[Callable[[Array], Array]] = None,
    ) -> "LinearScan":
        return self.replace(
            x(self.x) if x is not None else "unchanged",
            y(self.y) if y is not None else "unchanged",
            z(self.z) if z is not None else "unchanged",
        )

    def resize(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
    ) -> "LinearScan":
        if y is not None and self.y is None:
            raise ValueError("Cannot resize y because it is not defined on this scan")
        return self.replace(
            np.linspace(self.x[0], self.x[-1], x) if x is not None else "unchanged",
            np.linspace(self.y[0], self.y[-1], y) if y is not None else "unchanged",
            np.linspace(self.z[0], self.z[-1], z) if z is not None else "unchanged",
        )

    @property
    def axes(self) -> Tuple[Array, ...]:
        if self.y is not None:
            return self.x, self.y, self.z
        else:
            return self.x, self.z

    @property
    def cartesian_bounds(self):
        return self.bounds  # LinearScan is already in cartesian coordinates :)

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.CARTESIAN

    def __repr__(self):
        return f"LinearScan(<shape={self.shape}>)"


@overload
def linear_scan(x: Array, z: Array) -> LinearScan: ...  # 2D scan


@overload
def linear_scan(x: Array, y: Array, z: Array) -> LinearScan: ...  # 3D scan


def linear_scan(*xyz: Array) -> LinearScan:
    "Construct a linear scan. See LinearScan documentation for more details."
    x, y, z = parse_axes(xyz)
    return LinearScan(x, y, z)
