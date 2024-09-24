from vbeam.core import TransmittedWavefront, WaveData,ProbeGeometry
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.wavefront.plane import PlaneWavefront


@traceable_dataclass()
class FocusedSphericalWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        sender_source_dist = np.sqrt(np.sum((sender - wave_data.source) ** 2))
        source_point_dist = np.sqrt(np.sum((wave_data.source - point_position) ** 2))
        return sender_source_dist * np.sign(
            wave_data.source[2] - sender[2]
        ) - source_point_dist * np.sign(wave_data.source[2] - point_position[2])


@traceable_dataclass(("pw_margin",))
class FocusedHybridWavefront(TransmittedWavefront):
    pw_margin: float = 0.001

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        spherical_wavefront = FocusedSphericalWavefront()
        plane_wavefront = PlaneWavefront()
        wave_data.azimuth = np.arctan2(
            wave_data.source[0] - sender[0],
            wave_data.source[2] - sender[2],
        )
        wave_data.elevation = wave_data.source[1] - sender[1]
        return np.where(
            np.abs(point_position[2] - wave_data.source[2]) > self.pw_margin,
            spherical_wavefront(probe,sender, point_position, wave_data),
            plane_wavefront(probe,sender, point_position, wave_data),
        )
