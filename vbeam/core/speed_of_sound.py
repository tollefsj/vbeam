"""Interface for sampling the speed of sound on a line, typically the line between a
:term:`sender` and a :term:`point`, and a :term:`point` and a :term:`receiver`."""

from abc import abstractmethod

from vbeam.fastmath import Array
from vbeam.module import Module


class SpeedOfSound(Module):
    @abstractmethod
    def average(
        self,
        sender_position: Array,
        point_position: Array,
        receiver_position: Array,
    ) -> float:
        """Sample the speed of sound between the sender, the point position, and the
        receiver, and return the average.

        All positions are arrays of three elements: (x, y, z)."""
