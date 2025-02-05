# HESPI-Electron

# Instructions

1. `npm install`
2. `npm run start`

# Bundled Python
The app comes with its own bundled python environment.
You can call the bundled python environment directly as:

```
HESPI.app/Contents/Resources/python/hespi-env/bin/python3.10 -m typer HESPI.app/Contents/Resources/python/hespi/main.py --help 
```

The python version included is CPython 3.10 standalone binaries from: https://github.com/astral-sh/python-build-standalone/releases/tag/20250115
The environment includes the poetry environment used for development with the addition of HESPI's python code