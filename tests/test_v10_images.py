"""Offline encoding tests; no model calls or robot actions."""
import base64
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from astra.api_supervisor import image_block as legacy_image_block
from astra.chunk_supervisor import ChunkApiSupervisor, ChunkCodexSupervisor, ChunkMixin
from astra.v10_images import image_block


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'cache.jpg'

    def encode(self, size, fmt='PNG'):
        Image.new('RGB', size, (72, 112, 176)).save(self.path, format=fmt)
        original = self.path.read_bytes()
        block = image_block(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(block['image_url']['url'].startswith('data:image/jpeg;base64,'))
        sent = base64.b64decode(block['image_url']['url'].split(',', 1)[1])
        with Image.open(io.BytesIO(sent)) as image:
            self.assertEqual(image.format, 'JPEG')
            return image.size, original, sent

    def test_camera_png_in_jpg_cache_and_legacy_unchanged(self):
        size, original, sent = self.encode((1280, 720))
        self.assertEqual(size, (960, 540))
        legacy = legacy_image_block(self.path)['image_url']['url']
        self.assertTrue(legacy.startswith('data:image/png;base64,'))
        self.assertEqual(base64.b64decode(legacy.split(',', 1)[1]), original)

    def test_small_demo_jpeg_byte_identical(self):
        size, original, sent = self.encode((480, 360), 'JPEG')
        self.assertEqual(size, (480, 360))
        self.assertEqual(sent, original)

    def test_no_upscale_and_portrait_aspect_ratio(self):
        self.assertEqual(self.encode((320, 180))[0], (320, 180))
        self.assertEqual(self.encode((720, 1280))[0], (540, 960))
        self.assertEqual(self.encode((1920, 1080), 'JPEG')[0], (960, 540))

    def test_both_backends_share_attachment_preparation(self):
        self.assertIs(ChunkApiSupervisor.start, ChunkMixin.start)
        self.assertIs(ChunkCodexSupervisor.start, ChunkMixin.start)
        self.assertIs(ChunkMixin.start.__globals__['image_block'], image_block)


if __name__ == '__main__':
    unittest.main()
