import sys
import os
from pathlib import Path

if __name__ == "__main__":
    scripts_dir = Path(__file__).parent / "experiments_src" / "scripts"
    scripts = {}
    for it in scripts_dir.iterdir():
        if it.suffix != ".py" or it.name == "__init__.py":
            continue
        scripts[it.stem] = it.absolute()
    cmd = sys.argv[1]
    if cmd not in scripts:
        print(
            "Unknown command. To launch some builtin script try the following command:\n"
            "\tpython -m SeqNAS cmd --opt1 param --opt2 param ...\n"
            "where the `cmd` is one of the following:\n"
            f"\t{', '.join(scripts.keys())}."
        )

    os.execvp("python3", ["python3", scripts[cmd], *sys.argv[2:]])
