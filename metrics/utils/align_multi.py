import argparse 
import os 
import subprocess
import glob, re

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('--apix', type=float)
parser.add_argument('--org-vol', required=True)
parser.add_argument('--flip', action='store_true')
args = parser.parse_args()

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

file_pattern = "*.mrc"
files = glob.glob(os.path.join(args.org_vol, file_pattern))
gt_dir = sorted(files, key=natural_sort_key)

file_pattern = "*.mrc"
matching_files = glob.glob(os.path.join(args.dir, file_pattern))
matching_files = sorted(matching_files, key=natural_sort_key)

os.makedirs(os.path.join(args.dir, 'aligned'), exist_ok=True)
os.makedirs(os.path.join(args.dir, 'flipped_aligned'), exist_ok=True)

for i, file_path in enumerate(matching_files):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    new_filename = base_filename + ".mrc"
    destination_path = os.path.join(args.dir, "aligned", new_filename)
    ref_path = gt_dir[i]

    align_cmd = f"sbatch metrics/utils/align.slurm \
    {ref_path} \
    {os.path.join(args.dir, new_filename)} \
    {destination_path} \
    {os.path.join(args.dir, 'aligned', f'temp_{i:03d}.txt')}"


    print(align_cmd)
    subprocess.check_call(align_cmd, shell=True)
