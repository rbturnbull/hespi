import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--libsPaths", nargs='+', help="A list of paths where the python libs are installed")
args = parser.parse_args()

if args.libsPaths is not None:
    for p in args.libsPaths:
        sys.path.insert(0, p)
# sys.path.insert(0, "./hespi-libs")1
# sys.path.insert(0, "./hespi")
print(sys.path)

from main import app
app()

