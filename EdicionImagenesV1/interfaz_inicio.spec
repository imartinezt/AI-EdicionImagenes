# interfaz_inicio.spec
# -*- mode: python ; coding: utf-8 -*-

import os
import multiprocessing

block_cipher = None

project_dir = '/Users/imartinezt/Documents/GitHub/AI-EdicionImagenes/EdicionImagenesV1'
img_dir = os.path.join(project_dir, 'img')

a = Analysis(
    [os.path.join(project_dir, 'InterfazInicio.py')],
    pathex=[project_dir],
    binaries=[],
    datas=[
        (os.path.join(img_dir, 'Liverpool_logo.svg.png'), 'img'),
        (os.path.join(img_dir, 'loading.gif'), 'img')
    ],
    hiddenimports=['rembg', 'onnxruntime', 'onnxruntime_pybind11_state', 'multiprocessing'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='InterfazInicio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InterfazInicio'
)
