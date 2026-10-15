"""Extract operator demonstration frames without importing robot components."""
import hashlib
import math
from pathlib import Path


def prepare_reference_video(video_path, output_dir, frame_count=12):
    if type(frame_count) is not int or not 1 <= frame_count <= 32:
        raise ValueError('reference frame count must be between 1 and 32')
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'reference video not found: {source}')
    import cv2
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f'cannot open reference video: {source}')
        total_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(total_raw) or total_raw < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('reference video requires readable frames and a valid frame rate')
        total = int(total_raw)
        count = min(frame_count, total)
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f'cannot decode reference frame {index}')
            # Bound payload size for the existing SSH image transport.
            height, width = frame.shape[:2]
            if max(height, width) > 960:
                scale = 960 / max(height, width)
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            path = output / f'frame_{index:08d}.jpg'
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise OSError(f'cannot write reference frame: {path}')
            images.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'frame_index': index, 'time_seconds': index / fps})
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return {'source_path': str(source), 'source_sha256': digest.hexdigest(),
                'source_fps': fps, 'source_frame_count': total,
                'sampling': 'uniform_chronological_not_l2_annotations', 'images': images}
    finally:
        capture.release()
