# -*- mode: python ; coding: utf-8 -*-
import glob
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('paddleocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyclipper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('imghdr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('skimage')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('imgaug')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('imageio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy.io')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lmdb')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('paddle')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Bundle local model files so PaddleOCR won't try to download at runtime.
datas += [('whl', 'whl')]

# Paddle will import `paddle.utils.cpp_extension`, which imports setuptools/Cython at runtime.
# Cython requires non-.py utility source files (e.g. `Cython/Utility/CppSupport.cpp`) to exist.
tmp_ret = collect_all('Cython')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
datas += collect_data_files('Cython', includes=[
    'Utility/*',
    'Utility/**/*',
    'Utility/*.cpp',
    'Utility/*.h',
    'Utility/*.hpp',
    'Utility/*.pxd',
    'Utility/*.pyx',
])

tmp_ret = collect_all('setuptools')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
datas += collect_data_files('setuptools', includes=['_vendor/jaraco/text/*.txt'])

# Bundle local compiled extensions used by the server (built via `python setup.py build_ext --inplace`).
# PyInstaller may not discover them if excluded from Analysis.
# Note: when PyInstaller executes a spec, `__file__` may not be injected (depends on invocation),
# so fall back to current working directory.
_here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
for _pyd in glob.glob(os.path.join(_here, "treat_compare_img*.pyd")):
    binaries.append((_pyd, "."))
for _pyd in glob.glob(os.path.join(_here, "simplefem_focus*.pyd")):
    binaries.append((_pyd, "."))
if os.path.exists(os.path.join(_here, "simplefem_focus.py")):
    datas.append((os.path.join(_here, "simplefem_focus.py"), "."))
if os.path.exists(os.path.join(_here, "screenshot_lock.py")):
    datas.append((os.path.join(_here, "screenshot_lock.py"), "."))
hiddenimports += ["screenshot_lock"]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['treat_compare_img', 'simplefem_focus', 'server', 'ocr_detect'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ocrapp_pureray',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ocrapp_pureray',
)
