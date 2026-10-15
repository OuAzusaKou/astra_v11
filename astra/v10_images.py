"""Compress V10 model attachments only; leave captured files untouched."""
import base64
import io
from pathlib import Path

from PIL import Image


MAX_EDGE = 960
JPEG_QUALITY = 85


def image_block(path):
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in ('PNG', 'JPEG'):
            raise ValueError('unsupported V10 image format')
        # Small demo JPEGs are already compressed. Keep their exact bytes.
        if source.format != 'JPEG' or max(source.size) > MAX_EDGE:
            rgb = source.convert('RGB')
            rgb.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            rgb.save(output, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            raw = output.getvalue()
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')}}
