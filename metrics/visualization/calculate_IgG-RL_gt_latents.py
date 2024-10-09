import numpy as np 
import matplotlib.pyplot as plt 
from MDAnalysis.lib.distances import calc_dihedrals

import MDAnalysis as mda

 

file_root = 'IgG-RL/pdbs/'
n_structures = 100
pdb_list =  [''] * 100
for i in range(100):
    pdb_list [i] = file_root + f"{i:03}" + '.pdb'


pdb_list = ['']
universe = mda.Universe (pdb_list)
sel1 = universe.select_atoms('resid 1-213 and segid H or segid L')
hinge1 = universe.select_atoms('resid 213 and segid H and name CA') 
hinge2 = universe.select_atoms('resid 244 and segid H and name CA and resname CYS') 
sel2 = universe.select_atoms('not ( resid 1-213 and segid H or segid L)')
dihs = np.zeros(universe.trajectory.n_frames)
dists = np.zeros(universe.trajectory.n_frames)

for i in range(universe.trajectory.n_frames):
    universe.trajectory[i]
    coord0 = sel1.center_of_mass()
    coord1 = hinge1.center_of_mass()
    coord2 = hinge2.center_of_mass()
    coord3 = sel2.center_of_mass()
    dihs[i] = calc_dihedrals(coord0, coord1, coord2, coord3)
    dists[i] = np.linalg.norm((coord0, coord3))

dihs = np.array([angle + 2*np.pi if angle < 0 else angle for angle in dihs])
dihs = dihs/np.pi
write_array = np.array([dists,dihs]).T


#np.save('conf-het-2_CV_dihedral_distance.npy',write_array)
#$plt.xlabel('Center of mass distance d ($\AA$)',fontsize=20)
#plt.ylabel('Dihedral angle $\phi$ (radians/$\pi$)',fontsize=20)
plt.figure(figsize=(7,5))
plt.xlabel('Center of mass distance ($\AA$)',fontsize=20)
plt.ylabel('Dihedral angle (radians/ $\pi$)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0,2])
plt.scatter(dists,dihs,s=200,edgecolors='black',c = dists, cmap = 'viridis' )
plt.tight_layout()
plt.savefig('dihedral_distance.pdf',dpi=92)
plt.show()
