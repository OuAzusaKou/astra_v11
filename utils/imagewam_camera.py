"""Training-compatible single-frame H.264 roundtrip without FFmpeg subprocesses."""
import ctypes
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(str(Path(__file__).with_name('native') / 'imagewam_camera.so'))
    for name in ('imagewam_camera', 'imagewam_camera_bgr'):
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        function.restype = ctypes.c_int
    lib.imagewam_camera_init()
    return lib


def preprocess_rgb(frame, crop_percent=100, *, bgr=False):
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    output = np.empty((480, 640, 3), dtype=np.uint8)
    error = ctypes.create_string_buffer(512)
    function = _library().imagewam_camera_bgr if bgr else _library().imagewam_camera
    result = function(frame.ctypes.data, frame.shape[1], frame.shape[0],
                      crop_percent, output.ctypes.data, error, len(error))
    if result:
        raise RuntimeError(error.value.decode())
    return output


@lru_cache(maxsize=1)
def _workers():
    return ThreadPoolExecutor(max_workers=3, thread_name_prefix='imagewam_camera')


def preprocess_cameras(frames, crop=(100, 100, 100)):
    return list(_workers().map(preprocess_rgb, frames, crop))


def _resize_tile(image, size):
    import torch
    from torchvision.transforms import functional as transforms
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return transforms.resize(tensor, list(size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)


def _mosaic(tiles):
    import torch
    mosaic = torch.cat([tiles[0], torch.cat(tiles[1:], dim=-1)], dim=-2)
    return np.ascontiguousarray((mosaic * 2 - 1).clamp(-1, 1)[0].numpy())


_TILE_SIZES = ((224, 288), (112, 144), (112, 144))


def model_input(cameras):
    """Match standardized_native_288 float32 antialiased training mosaic."""
    return _mosaic([_resize_tile(image, size) for image, size in zip(cameras, _TILE_SIZES)])


def _bgr_tile(frame, crop, size):
    return _resize_tile(preprocess_rgb(frame, crop, bgr=True), size)


def prepare_model_input(frames, crop=(100, 100, 100)):
    """Process BGR cameras through resize in parallel, then assemble the mosaic."""
    return _mosaic(list(_workers().map(_bgr_tile, frames, crop, _TILE_SIZES)))
