"""Build the in-process FFmpeg camera preprocessor against system FFmpeg."""
from pathlib import Path
import subprocess
import argparse
from ctypes.util import find_library

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include', help='Additional include directory for extracted FFmpeg development headers.')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', 'libavcodec', 'libavutil', 'libswscale'], text=True).split()
flags.append('-l:' + find_library('avfilter'))
if args.include:
    flags.append('-I' + args.include)
output = root / 'utils/native/imagewam_camera.so'
temporary = output.with_suffix('.so.tmp')
subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC', str(output.with_suffix('.cpp')), '-o', str(temporary), *flags], check=True)
temporary.replace(output)
print(output)
