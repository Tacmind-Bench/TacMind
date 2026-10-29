"""Export GSmini RGB, Marker dots, Deformation arrows and native forces.

Drawing adapted from UniVTAC TacEx 0dafa102 and Tabero 56a3b905:
their simulation_approaches/fots/fots_marker_sim.py drawing functions.
License: Apache-2.0 (benchmark/backends/gsmini/tacex/LICENSE.Tabero).
"""

import argparse
from contextlib import redirect_stdout, suppress
import json
from pathlib import Path
from stat import S_IMODE
import sys
from tempfile import TemporaryDirectory

import h5py

import hdf5plugin  # noqa: F401 -- register lossless recording filters
import numpy as np

from benchmark.io.video import rgb_video
from benchmark.backends.gsmini.markers import _draw_markers
from benchmark.io.image_storage import rgb_sequence

SIDES = ('left', 'right')
VIEWS = ('rgb', 'marker', 'deformation')
TILE_SIZE = (320, 240)
RGB_BATCH_FRAMES = 8
CHANNELS = ('normal_fx', 'normal_fy', 'normal_fz', 'tangent_fx', 'tangent_fy', 'tangent_fz')



def _read(observations, metadata):
    contact = metadata['gsmini_contact']
    dt, physics_dt = metadata['control_dt'], contact['physics_dt']
    images = [rgb_sequence(observations[f'gsmini_{side}']) for side in SIDES]
    markers = [observations[f'gsmini_{side}_marker_motion'][:] for side in SIDES]
    normal = observations['gsmini_gel_target_normal_force_w'][:]
    tangent = observations['gsmini_gel_target_tangent_force_w'][:]
    times = observations['gsmini_contact_time'][:, -1]
    steps = np.rint(times/physics_dt).astype(np.int64)
    targets = np.concatenate((normal, tangent), axis=-1).astype(np.float64)
    forces = dict(time_s=times, physics_step=steps, target_force6_w=targets,
                  force6_w=targets.sum(axis=2, dtype=np.float64))
    return images, markers, forces, dt, contact['filter_paths']


def export_episode(source, output):
    """Publish the fixed combined preview, native force arrays and their metadata."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(source, 'r') as dataset:
        episode = dataset['data/demo_0']
        metadata = json.loads(dataset['data'].attrs['env_args'])
        images, markers, forces, dt, filters = _read(episode['obs'], metadata)
        count = len(forces['time_s'])
        width, height = TILE_SIZE[0]*len(SIDES), TILE_SIZE[1]*len(VIEWS)
        output.mkdir(parents=True)
        try:
            with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as working:
                working = Path(working)
                path = working/'tactile.mp4'
                canvas = np.empty((height, width, 3), dtype=np.uint8)
                with rgb_video(path, width=width, height=height, fps=1/dt,
                               encoder_threads=2, buffered=True) as write:
                    # Read bounded groups aligned with recorder RGB chunks to
                    # amortize HDF5 selection and allocation overhead.
                    for first in range(0, count, RGB_BATCH_FRAMES):
                        batches = [image[first:first+RGB_BATCH_FRAMES] for image in images]
                        for offset in range(len(batches[0])):
                            index = first+offset
                            for side, batch in enumerate(batches):
                                tiles = canvas[:, side*TILE_SIZE[0]:(side+1)*TILE_SIZE[0]].reshape(
                                    len(VIEWS), TILE_SIZE[1], TILE_SIZE[0], 3)
                                tiles[:] = batch[offset]
                                _draw_markers(tiles[1], tiles[2], markers[side][index])
                            write(canvas)
                np.savez_compressed(working/'contact_forces.npz', **forces)
                stream = dict(width=width, height=height, nb_frames=count, fps=1/dt)
                report = dict(format='tacmind.gsmini.tactile-videos', source=str(source.resolve()), frame_count=count,
                    observation_group='data/demo_0/obs', observation_clock='pre-action',
                    time_range_s=[float(forces['time_s'][0]), float(forces['time_s'][-1])],
                    layout=dict(rows=list(VIEWS), pad_order=list(SIDES), tile_size=list(TILE_SIZE)),
                    markers=dict(color='black', coordinates='initial,current XY pixels',
                        label='Marker', style='TacEx dots', radius_px=3,
                        rendering='10x supersampled circle; Gaussian blur (17x17, sigma 15); cubic downsample',
                        position='current + 0.5 pixels; 0.1-pixel template bins',
                        source=metadata.get('gsmini_marker_motion'),
                        rgb='raw Taxim; coordinates and raw RGB are unchanged'),
                    deformation=dict(label='Deformation', style='NumPy displacement arrows', color='black',
                        arrow_origin='initial marker position', arrow_endpoint='current + (current - initial)',
                        coordinate_quantization='truncate initial/current XY to integers before drawing',
                        arrow_scale=1, thickness_px=2, tip_length=.2,
                        overlay='multiply tactile RGB by normalized arrow mask / 255'),
                    forces=dict(file='contact_forces.npz', array='force6_w', shape=list(forces['force6_w'].shape),
                        target_array='target_force6_w', target_shape=list(forces['target_force6_w'].shape),
                        channels=list(CHANNELS), units='N', frame='world', pad_order=list(SIDES),
                        filter_paths=filters, time_array='time_s', physics_step_array='physics_step',
                        source=['obs/gsmini_gel_target_normal_force_w', 'obs/gsmini_gel_target_tangent_force_w'],
                        decomposition='native_contact_normal_and_friction',
                        semantics='native contact normal and friction force acting on gel pads; not force/torque',
                        aggregation='sum of configured target filters; excludes housing and unfiltered contacts'),
                    videos={path.name: stream})
                (working/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
                working.chmod(S_IMODE(output.stat().st_mode))
                working.rename(output)
        except BaseException:
            with suppress(OSError):
                output.rmdir()
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='GSmini HDF5 with native six-dimensional forces')
    parser.add_argument('--output', type=Path, required=True, help='New directory for videos, contact forces and metadata')
    parser.add_argument('--all-views', action='store_true', help='Include agent/wrist views, individual tactile views and a synchronized overview')
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        if args.all_views:
            from benchmark.backends.gsmini.video_views import export_all_views
            report = export_all_views(args.input, args.output)
        else:
            report = export_episode(args.input, args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
