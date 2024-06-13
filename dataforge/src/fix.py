import os
import numpy as np

from typing import Optional, List
from itertools import permutations
from .generic import append_suffix_to_filename


class Mol:
    def __init__(self, elements, coords, atypes, bond_idcs) -> None:
        self.elements = np.array(elements)
        self.coords = np.stack(coords, axis=0)
        self.atypes = np.array(atypes)
        self.bond_idcs = [np.array(bid) for bid in bond_idcs]
    
    def rows(self):
        rows = []
        for e, c, a, b in zip(self.elements, self.coords, self.atypes, self.bond_idcs):
            row = f"{e}       {'       '.join(['{:.8f}'.format(x) for x in c])} {a} {' '.join([str(x) for x in b])}\n"
            rows.append(row)
        return rows
    
    def get_wrong_bonds(self, coords: np.ndarray, bond_idcs: List[np.ndarray]):
        wrong_indices = []
        wrong_atypes = []
        for src_id, trg_idcs in enumerate(bond_idcs):
            bond_lengths = np.linalg.norm(coords[trg_idcs] - coords[src_id], axis=-1)
            if np.any(bond_lengths > 1.9):
                wrong_indices.append(src_id)
                wrong_atypes.append(self.atypes[src_id])
        return np.array(wrong_indices), np.array(wrong_atypes)
    
    def fix_symmetric_atoms_coords(self, coords: Optional[np.ndarray] = None, bond_idcs: Optional[List[np.ndarray]] = None, level: int=0, level_max: int=1):
        if coords is None:
            coords = self.coords
        if bond_idcs is None:
            bond_idcs = self.bond_idcs
        wrong_indices, wrong_atypes = self.get_wrong_bonds(coords, bond_idcs)
        if len(wrong_atypes) == 0:
            self.coords = coords
            return True
        if level == level_max:
            return False
        unique_elements, counts = np.unique(wrong_atypes, return_counts=True)
        duplicate_elements = unique_elements[counts > 1]
        for element in duplicate_elements:
            starting_indices = wrong_indices[wrong_atypes == element]
            for new_indices in list(permutations(starting_indices))[1:]:
                new_coords = np.copy(coords)
                new_indices = np.array(list(new_indices))
                new_coords[starting_indices] = new_coords[new_indices]
                to_swap_idcs = []
                for si in starting_indices:
                    to_swap_idcs.append(self.bond_idcs[si][~np.isin(self.bond_idcs[si], wrong_indices)])
                for id1, id2 in zip(to_swap_idcs[0], to_swap_idcs[1]):
                    temp = np.copy(new_coords[id1])
                    new_coords[id1] = new_coords[id2]
                    new_coords[id2] = temp
                is_fixed = self.fix_symmetric_atoms_coords(new_coords, level=level+1, level_max=len(duplicate_elements))
                if is_fixed:
                    return True
        return False


def fix_single_mol(elements, coords, atypes, bond_idcs) -> Mol:
    mol = Mol(elements, coords, atypes, bond_idcs)
    if mol.fix_symmetric_atoms_coords():
        return mol
    raise Exception

def fix_bonds(filename: str):
    natoms = 0
    filename_fixed = append_suffix_to_filename(filename, "_fixed_bonds")
    with open(filename_fixed, 'w') as f_out:
        with open(filename, 'r') as f_in:
            flag = 'natoms'
            for line in f_in.readlines():
                if flag == 'natoms':
                    natoms = line
                    flag = 'energy'
                elif flag == 'energy':
                    energies = line
                    flag = 'xyz'
                    elements = []
                    coords = []
                    atypes = []
                    bond_idcs = []
                else:
                    splits = line.split()
                    if len(splits) == 1:
                        mol = fix_single_mol(elements, coords, atypes, bond_idcs)
                        lines = [
                            str(natoms),
                            energies,
                            *mol.rows(),
                        ]
                        f_out.writelines(lines)
                        natoms = line
                        flag = 'energy'
                    else:
                        elements.append(splits[0])
                        coords.append(np.array([float(x) for x in splits[1:4]]))
                        atypes.append(splits[4])
                        bond_idcs.append([int(x) for x in splits[5:]])
            mol = fix_single_mol(elements, coords, atypes, bond_idcs)
            lines = [
                str(natoms),
                energies,
                *mol.rows(),
            ]
            f_out.writelines(lines)
    os.replace(filename_fixed, filename)