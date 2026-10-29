"""Synchronized camera and tactile videos for a recorded episode."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import h5py
import hdf5plugin
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from benchmark.backends.gsmini.export import _read, export_episode
from benchmark.backends.gsmini.markers import _draw_markers
from benchmark.io.image_storage import rgb_sequence
from benchmark.io.video import rgb_video


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def _export_views(source, output):
    source, output = (Path(source).resolve(), Path(output).resolve())
    with h5py.File(source, 'r') as f:
        episode = f['data/demo_0']
        meta = json.loads(f['data'].attrs['env_args'])
        tactile, markers, forces, dt, _ = _read(episode['obs'], meta)
        cameras = {name: rgb_sequence(episode['obs'][name]) for name in ('agentview', 'wristview')}
        count = len(tactile[0])
        output.mkdir(parents=True)
        native = output / 'native'
        export_episode(source, native)
        for name in ('tactile.mp4', 'contact_forces.npz', 'metadata.json'):
            (native / name).rename(output / ('tactile_metadata.json' if name == 'metadata.json' else name))
        native.rmdir()
        normal = np.linalg.norm(forces['force6_w'][:, :, :3], axis=-1)
        tangent = np.linalg.norm(forces['force6_w'][:, :, 3:], axis=-1)
        contact_index = int(normal.sum(axis=1).argmax())
        width, height = (1168, 1130)
        base = Image.new('RGB', (width, height), (18, 24, 32))
        draw = ImageDraw.Draw(base)
        font, title_font = _font(16), _font(23)
        colors = ((58, 199, 182), (248, 173, 80))
        title = f"GSmini / {meta['suite'].removeprefix('tac_').upper()} {meta['task_id']:02d} / {meta.get('task_definition', {}).get('task_name', '')} / seed {episode.attrs['seed']}"
        draw.text((16, 14), title, font=title_font, fill=(236, 242, 248))
        draw.text((16, 46), 'Recorded episode  |  synchronized observations  |  original speed', font=font, fill=(153, 170, 188))
        for label, y in (('AGENT VIEW', 78), ('WRIST VIEW', 604)):
            draw.rectangle((16, y, 496, y + 29), fill=(32, 42, 54))
            draw.text((26, y + 4), label, font=font, fill=(222, 232, 242))
        for label, y in (('TACTILE RGB', 78), ('MARKER / TacEx dots', 358), ('DEFORMATION / Tabero', 638)):
            draw.rectangle((512, y, 1152, y + 27), fill=(32, 42, 54))
            draw.text((522, y + 3), label, font=font, fill=(222, 232, 242))
            draw.text((797, y + 3), 'LEFT', font=font, fill=colors[0])
            draw.text((1093, y + 3), 'RIGHT', font=font, fill=colors[1])
        draw.text((522, 918), 'NATIVE CONTACT FORCE  |  world frame  |  N', font=font, fill=(222, 232, 242))
        plots = []
        for values, label, top in ((normal, 'Normal', 950), (tangent, 'Friction', 1032)):
            left, right, bottom = (594, 1140, top + 64)
            limit = max(float(values.max()) * 1.08, 0.01)
            draw.text((522, top), label, font=font, fill=(153, 170, 188))
            draw.text((522, top + 23), f'{limit:.2f} N', font=font, fill=(153, 170, 188))
            draw.rectangle((left, top, right, bottom), fill=(27, 35, 46))
            for fraction in (0, 0.5, 1):
                y = round(bottom - fraction * (bottom - top))
                draw.line((left, y, right, y), fill=(48, 59, 73))
            for side, color in enumerate(colors):
                points = [(round(left + i * (right - left) / max(count - 1, 1)),
                           round(bottom - float(v) / limit * (bottom - top)))
                          for i, v in enumerate(values[:, side])]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
            plots.append((left, top, right, bottom))
        template = np.array(base)
        with ExitStack() as stack:
            writers = {}
            for name, images in cameras.items():
                writers[name] = stack.enter_context(rgb_video(output / (name + '.mp4'),
                    width=images.shape[2], height=images.shape[1], fps=1 / dt,
                    encoder_threads=1, buffered=True))
            for side in ('left', 'right'):
                for kind in ('rgb', 'marker', 'deformation'):
                    name = f'tactile_{side}_{kind}'
                    writers[name] = stack.enter_context(rgb_video(output / (name + '.mp4'),
                        width=320, height=240, fps=1 / dt, encoder_threads=1, buffered=True))
            overview = stack.enter_context(rgb_video(output / 'overview.mp4',
                width=width, height=height, fps=1 / dt, encoder_threads=2, buffered=True))
            for first in range(0, count, 8):
                rgb_batches = {name: seq[first:first + 8] for name, seq in cameras.items()}
                touch_batches = [seq[first:first + 8] for seq in tactile]
                for offset in range(len(touch_batches[0])):
                    index = first + offset
                    frame = template.copy()
                    for name, top in (('agentview', 108), ('wristview', 634)):
                        image = rgb_batches[name][offset]
                        h, w = image.shape[:2]
                        scale = min(1., 480 / max(h, w))
                        tile = cv2.resize(image, (round(w*scale), round(h*scale))) if scale < 1 else image
                        h, w = tile.shape[:2]
                        y, x = (top + (480 - h) // 2, 16 + (480 - w) // 2)
                        frame[y:y + h, x:x + w] = tile
                        writers[name](image)
                    for side, label in enumerate(('left', 'right')):
                        x = 512 + 320 * side
                        image = touch_batches[side][offset]
                        frame[106:346, x:x + 320] = image
                        dots = frame[386:626, x:x + 320]
                        arrows = frame[666:906, x:x + 320]
                        dots[:] = image
                        arrows[:] = image
                        _draw_markers(dots, arrows, markers[side][index])
                        for kind, tile in zip(('rgb', 'marker', 'deformation'), (image, dots, arrows)):
                            writers[f'tactile_{label}_{kind}'](tile)
                    for left, top, right, bottom in plots:
                        x = round(left + index * (right - left) / max(count - 1, 1))
                        cv2.line(frame, (x, top), (x, bottom), (235, 240, 248), 1)
                    canvas = Image.fromarray(frame)
                    draw = ImageDraw.Draw(canvas)
                    draw.text((873, 17), f'{index * dt:05.2f} / {count * dt:.2f} s', font=font, fill=(236, 242, 248))
                    overview(np.asarray(canvas))
                    if index == contact_index:
                        canvas.save(output / 'contact_frame.png')
        report = dict(
            suite=meta['suite'], task_id=meta['task_id'],
            seed=int(episode.attrs['seed']), task_name=meta.get('task_definition', {}).get('task_name'),
            source=str(source), frame_count=count,
            control_dt=dt, duration_s=count * dt, cameras=meta.get('cameras', {}),
            videos=[path.name for path in sorted(output.glob('*.mp4'))],
            marker='TacEx dots', deformation='Tabero NumPy arrows', observation_clock='pre-action')
        (output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def export_all_views(source, output):
    """Write ten videos at the recorded rate, publishing only a complete export."""
    source, output = (Path(source).resolve(), Path(output).resolve())
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        working = Path(temporary) / 'views'
        report = _export_views(source, working)
        working.rename(output)
    return report
