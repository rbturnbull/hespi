import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--libsPaths", action='append', help="A list of paths where the python libs are installed")

# Only parse the arguments defined here, and leave the rest for Typer in HESPI
args, remaining = parser.parse_known_args()
print(f"run_hespi args: {args}\nRemaining Typer args: {remaining}\n")
sys.argv = sys.argv[:1]+remaining # Make sure to remove this script's arguments
print(f"ARGS: {args}\nRemaining: {remaining}\nTyper Args: {sys.argv}")

if args.libsPaths is not None:
    for p in args.libsPaths:
        sys.path.insert(0, p)
# sys.path.insert(0, "./hespi-libs")1
# sys.path.insert(0, "./hespi")
print(f"Lib Paths: {sys.path}")


from main import app
app()

