import os
import argparse
import subprocess
import numpy as np
from cryodrgn import mrcfile

CHUNK = 10000 # restart ChimeraX session periodically to avoid OOM errors

def parse_args():
    parser = argparse.ArgumentParser(description='Generate mrc volumes from atomic model trajectory')
    parser.add_argument('pdb', help='Path to seed PDB file')
    parser.add_argument('traj', help='Path to trajectory file')
    parser.add_argument('num_models', type=int, help='Number of structures in the trajectory to generate volumes for')
    parser.add_argument('--Apix', type=float, default=1.5, help='Pixel size of volumes')
    parser.add_argument('-D', type=int, default=256, help='Box size of volumes')
    parser.add_argument('--res', type=float, default=3.0, help='Resolution to simulate density')
    parser.add_argument('-c', required=True, help='Path to ChimeraX binary, e.g. ~/chimerax-1.6.1/bin/ChimeraX')
    parser.add_argument('-o', required=True, help='Path to directory where volumes will be stored')
    return parser.parse_args()


class CXCFile:
    def __init__(self):
        self.commands = []

    def add(self, command: str):
        self.commands.append(command)

    def save(self, file_path: os.path.abspath):
        with open(file_path, "w") as file:
            file.writelines('\n'.join(self.commands))

    def execute(self, chimerax_path: os.path.abspath, cxc_path: os.path.abspath):
        self.save(cxc_path)
        chimerax_command = [chimerax_path, "--nogui", "--cmd", f"open {cxc_path}"]
        try:
            subprocess.run(chimerax_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        os.remove(cxc_path)


def pad_vol(path, Apix, D):
    data, header = mrcfile.parse_mrc(path)
    x,y,z = data.shape    
    new_data = np.zeros((D,D,D), dtype=np.float32)    
    i, j, k = (D-x)//2, (D-y)//2, (D-z)//2
    new_data[i:(i+x),j:(j+y),k:(k+z)] = data
    orig_x, orig_y, orig_z = header.origin
    new_header = mrcfile.get_mrc_header(
        new_data, True, 
        Apix=Apix, 
        xorg=(orig_x-k*Apix), 
        yorg=(orig_y-j*Apix), 
        zorg=(orig_z-i*Apix)
    )
    mrcfile.write_mrc(path, new_data, new_header)


def center_all_vols(num_models, outdir):
    for i in range(num_models):
        path = os.path.join(outdir, f'vol_{i:05d}.mrc')
        data, header = mrcfile.parse_mrc(path)
        header.origin = (0., 0., 0.)
        mrcfile.write_mrc(path, data, header)


def generate_ref_vol(pdb_path, outdir, chimerax_path, res, Apix, D):
    cxc = CXCFile()
    cxc.add(f"open {os.path.abspath(pdb_path)}")
    cxc.add(f"molmap #1 {res} gridSpacing {Apix}")
    cxc.add(f"save {os.path.abspath(os.path.join(outdir, 'ref.mrc'))} #2")
    cxc.add("exit")
    cxc.execute(chimerax_path, os.path.abspath(os.path.join(outdir, 'commands.cxc')))
    pad_vol(os.path.abspath(os.path.join(outdir, 'ref.mrc')), Apix, D)


def generate_all_vols(pdb_path, traj_path, num_models, outdir, chimerax_path, res, Apix):
    for start in range(0, num_models, CHUNK):
        cxc = CXCFile()
        cxc.add(f"open {os.path.abspath(pdb_path)}")
        cxc.add(f"open {os.path.abspath(traj_path)}")
        cxc.add(f"open {os.path.abspath(os.path.join(outdir, 'ref.mrc'))}")
        for i in range(start, min(start+CHUNK, num_models)):
            cxc.add(f"coordset #1 {i+1}")
            cxc.add(f"molmap #1 {res} gridSpacing {Apix}")
            cxc.add("vol resample #3 onGrid #2")
            cxc.add(f"save {os.path.abspath(os.path.join(outdir, f'vol_{i:05d}.mrc'))} #4")
            cxc.add("close #3-4")
        cxc.add("exit")
        cxc.execute(chimerax_path, os.path.abspath(os.path.join(outdir, 'commands.cxc')))
    center_all_vols(num_models, outdir)
    os.remove(os.path.join(outdir, 'ref.mrc'))


if __name__=="__main__":
    args = parse_args()
    os.makedirs(args.o)
    generate_ref_vol(args.pdb, args.o, args.c, args.res, args.Apix, args.D)
    generate_all_vols(args.pdb, args.traj, args.num_models, args.o, args.c, args.res, args.Apix)
