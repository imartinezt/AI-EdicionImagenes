from setuptools import setup

APP = ['InterfazInicio.py']
DATA_FILES = [
    'Liverpool_logo.svg.png',
    'loading.gif',
    'keys.json'
]
OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'PIL', 'asyncio', 'tkinter', 'torch', 'transformers', 'google.cloud', 'google.oauth2', 'rembg', 'aiofiles'
    ],
    'includes': [
        'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.UnidentifiedImageError', 'PIL.ImageSequence',
        'asyncio', 'tkinter', 'torch', 'transformers', 'google.cloud', 'google.oauth2', 'rembg', 'aiofiles'
    ],
    'excludes': ['tkinter.test', 'tkinter.tix', 'tkinter.scrolledtext']
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app', 'Pillow', 'torch', 'transformers', 'google-cloud-vision', 'rembg', 'aiofiles'],
)
