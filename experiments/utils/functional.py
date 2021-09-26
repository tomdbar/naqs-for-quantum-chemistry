import os

from src.utils.system import mk_dir

def export_script(script_fname, target_dir):
    mk_dir(target_dir, quiet=True)
    target_fname = os.path.join(target_dir, os.path.split(script_fname)[-1])
    with open(script_fname, 'r') as f:
        with open(target_fname, 'w') as out:
            for line in f.readlines():
                out.write(line)

def export_summary(fname, content):
    mk_dir(os.path.dirname(fname), quiet=True)
    with open(fname, "w") as f:
        if type(content) is list:
            f.write('\n'.join(content))
        else:
            f.write(content)