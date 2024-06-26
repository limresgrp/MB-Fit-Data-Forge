import operator
import numpy as np
from typing import Optional, Union, List, Tuple
from functools import reduce
from itertools import product
from itertools import chain
from dataforge.src import get_bonds, get_angles, get_dihedrals, union_rows_2d, intersect_rows_2d


class Monomer:

    id: int
    name: str
    heavy_atoms_names: np.ndarray
    heavy_atoms_idcs: np.ndarray
    orig_all_atoms_idcs: np.ndarray       # indices of all atoms belonging and connected to the monomer
    orig_atoms_idcs: np.ndarray           # indices of all atoms belonging to the monomer
    orig_heavy_atoms_idcs: np.ndarray     # indices of heavy atoms belonging to the monomer
    orig_connected_atoms_idcs: np.ndarray # indices of heavy atoms not belonging to the monomer

    orig_bond_idcs: np.ndarray
    orig_angle_idcs: np.ndarray
    orig_dihedral_idcs: np.ndarray

    bond_names: np.ndarray = None
    angle_names: np.ndarray = None
    dihedral_names: np.ndarray = None

    bond_values: Optional[np.ndarray] = None
    angle_values: Optional[np.ndarray] = None
    dihedral_values: Optional[np.ndarray] = None
    

    def __init__(
        self,
        id: int,
        name: str,
        heavy_atoms_names: Union[str, List[str]],
        heavy_atoms_idcs: Union[int, Tuple[int]],
        
        orig_all_idcs: np.ndarray,
        
        orig_all_bond_idcs: np.ndarray,
        orig_all_angle_idcs: np.ndarray,
        orig_all_dihedral_idcs: np.ndarray,

        orig_all_atom_names: np.ndarray,
        orig_all_atom_types: np.ndarray,
    ) -> None:
        
        self.id = id
        self.name = name

        self.heavy_atoms_names = np.array(heavy_atoms_names)
        self.heavy_atoms_idcs = np.array(heavy_atoms_idcs)
        self.orig_heavy_atoms_idcs = orig_all_idcs[self.heavy_atoms_idcs]

        self.orig_bond_idcs = orig_all_bond_idcs[np.any(np.isin(orig_all_bond_idcs, self.orig_heavy_atoms_idcs), axis=1)]
        self.orig_all_atoms_idcs = np.unique(self.orig_bond_idcs)

        self.orig_connected_atoms_idcs = self.orig_all_atoms_idcs[np.logical_and(
            ~np.isin(
                self.orig_all_atoms_idcs, self.orig_heavy_atoms_idcs
            ),
            orig_all_atom_types[self.orig_all_atoms_idcs] != 'H',
        )]
        self.orig_atoms_idcs = np.setdiff1d(
            self.orig_all_atoms_idcs,            # take all atoms connected and belonging to monomer
            self.orig_connected_atoms_idcs,      # exclude the heavy atoms connected to the monomer
            assume_unique=True,
        )
        
        self.orig_angle_idcs = orig_all_angle_idcs[np.all(np.isin(orig_all_angle_idcs, self.orig_all_atoms_idcs), axis=1)]
        self.orig_dihedral_idcs = orig_all_dihedral_idcs[np.all(np.isin(orig_all_dihedral_idcs, self.orig_all_atoms_idcs), axis=1)]

        self.bond_names = np.array(
            [f"{orig_all_atom_names[b0]}-{orig_all_atom_names[b1]}"
             for b0, b1
             in self.orig_bond_idcs]
            )
        self.angle_names = np.array(
            [f"{orig_all_atom_names[a0]}-{orig_all_atom_names[a1]}-{orig_all_atom_names[a2]}"
             for a0, a1, a2
             in self.orig_angle_idcs]
        )
        self.dihedral_names = np.array(
            [f"{orig_all_atom_names[d0]}-{orig_all_atom_names[d1]}-{orig_all_atom_names[d2]}-{orig_all_atom_names[d3]}"
             for d0, d1, d2, d3
             in self.orig_dihedral_idcs]
        )

    def compute_descriptors(self, orig_all_pos: np.ndarray):
        self.bond_values = get_bonds(orig_all_pos, self.orig_bond_idcs)
        self.angle_values = get_angles(orig_all_pos, self.orig_angle_idcs)
        self.dihedral_values = get_dihedrals(orig_all_pos, self.orig_dihedral_idcs)
    
    @property
    def descriptor_values(self) -> np.ndarray:
        if self.bond_values is None:
            raise Exception("Compute descriptors first!")
        return np.concatenate([self.bond_values, self.angle_values, self.dihedral_values], axis=1)

    @property
    def descriptor_names(self) -> np.ndarray:
        return np.concatenate(
            [self.bond_names, self.angle_names, self.dihedral_names]
        )


class Multimer:

    name: str
    monomers: List[Monomer]
    monomers_idcs: List[int]

    orig_all_atoms_idcs: np.ndarray       # indices of all atoms belonging and connected to the multimer
    orig_atoms_idcs: np.ndarray           # indices of all atoms belonging to the multimer
    orig_heavy_atoms_idcs: np.ndarray     # indices of heavy atoms belonging to the multimer
    orig_connected_atoms_idcs: np.ndarray # indices of heavy atoms not belonging to the multimer
    
    orig_bond_idcs: np.ndarray
    orig_angle_idcs: np.ndarray
    orig_dihedral_idcs: np.ndarray

    bond_values: Optional[np.ndarray] = None
    angle_values: Optional[np.ndarray] = None
    dihedral_values: Optional[np.ndarray] = None
    bond_names: Optional[np.ndarray] = None
    angle_names: Optional[np.ndarray] = None
    dihedral_names: Optional[np.ndarray] = None

    def __init__(
        self,
        monomers: List[Monomer],
        orig_all_atom_types: np.ndarray,
    ) -> None:
        
        # self.monomers is a sorted list of Monomer objects
        # The sorting is done based on the name, and in case of the same name, it is done based on the ID.
        self.monomers = sorted(monomers, key=operator.attrgetter('name', 'id'))
        del monomers
        self.monomers_idcs = [str(m.id) for m in self.monomers]
        
        # -------------------------------- ! ! ! ---------------------------------- #
        # orig_ prefix refers to a property of the original database.
        # If it is not present, the property is relative to the multimer 
        # (e.g. idcs always start from 0 relative to multimer,
        # but could start and have any value if they belong to the original dataset)
        # -------------------------------- ! ! ! ---------------------------------- #

        # All heavy atoms of the multimer (no H)
        self.orig_heavy_atoms_idcs = np.concatenate([m.orig_heavy_atoms_idcs for m in self.monomers])
        
        self.orig_heavy_atoms_monomer_names = list(chain.from_iterable([[m.name for _ in m.orig_heavy_atoms_idcs] for m in self.monomers]))
        self.orig_heavy_atoms_idcs_to_monomer_names = {
            k: v
            for k, v in zip(self.orig_heavy_atoms_idcs, self.orig_heavy_atoms_monomer_names)
        }

        self.orig_bond_idcs = reduce(union_rows_2d, ([m.orig_bond_idcs for m in self.monomers]))
        self.orig_angle_idcs = reduce(union_rows_2d, ([m.orig_angle_idcs for m in self.monomers]))
        self.orig_dihedral_idcs = reduce(union_rows_2d, ([m.orig_dihedral_idcs for m in self.monomers]))

        # All atoms connected + belonging to the multimer (heavy + H)
        self.orig_all_atoms_idcs = np.unique(self.orig_bond_idcs)
        # All (heavy) atoms connected, but not belonging to the multimer
        self.orig_connected_atoms_idcs = self.orig_all_atoms_idcs[np.logical_and(
            ~np.isin(
                self.orig_all_atoms_idcs, self.orig_heavy_atoms_idcs
            ),
            orig_all_atom_types[self.orig_all_atoms_idcs] != 'H',
        )]
        # All atoms of the multimer (heavy + H)
        self.orig_atoms_idcs = np.setdiff1d(
            self.orig_all_atoms_idcs,            # take all atoms connected and belonging to multimer
            self.orig_connected_atoms_idcs,      # exclude the heavy atoms connected to the multimer
            assume_unique=True,
        )

        # --- ! ! ! A T T E N T I O N ! ! ! --- #

        # Multimer name property is used to distinguish different Multimers.
        
        # This algorithm is not able to distinguish multimers of degree 4+ that differ in the
        # connectivity among monomers with the same name
        
        monomer_names = [mon.name for mon in self.monomers]
        _, monomer_name_unique_idcs = np.unique(np.array(monomer_names), return_inverse = True)
        monomer_name_unique_idcs = [str(x) for x in monomer_name_unique_idcs]
        monomer_name_unique_to_monomer_name_unique_idcs = {
            k: v
            for k, v in zip(monomer_names, monomer_name_unique_idcs)
        }
        self.orig_heavy_atoms_idcs_to_monomer_name_unique_idcs = {
            k: monomer_name_unique_to_monomer_name_unique_idcs[v]
            for k, v in zip(self.orig_heavy_atoms_idcs, self.orig_heavy_atoms_monomer_names)
        }
        self.name = '.'.join(monomer_names)
        
        if len(self.monomers) > 2:
            name = []
            for m, name_unique_id in zip(self.monomers, monomer_name_unique_idcs):
                connections = []
                other_monomers_all_orig_heavy_atoms_idcs = np.array(list(set(self.orig_heavy_atoms_idcs).union(set(self.orig_connected_atoms_idcs)) - set(m.orig_heavy_atoms_idcs)))
                for pair in product(m.orig_heavy_atoms_idcs, other_monomers_all_orig_heavy_atoms_idcs):
                    pair = np.array(pair)
                    if len(intersect_rows_2d(self.orig_bond_idcs, pair)>0) or len(intersect_rows_2d(self.orig_bond_idcs, np.ascontiguousarray(pair[::-1]))>0):
                        connections.append(self.orig_heavy_atoms_idcs_to_monomer_name_unique_idcs.get(pair[1], 'H'))
                name.append(f"{name_unique_id}_{''.join(sorted(connections))}")
            self.name = f"{self.name}|{'.'.join(sorted(name))}"

        # ------------------------------------- #
        

    def compute_descriptors(self):
        bond_values, angle_values, dihedral_values = [], [], []
        bond_names, angle_names, dihedral_names = [], [], []
        already_computed_bond, already_computed_angle, already_computed_dihedral = [], [], []
        for monomer in self.monomers:
            for bond_idcs, bond_value, bond_name in zip(
                monomer.orig_bond_idcs,
                monomer.bond_values.T,
                monomer.bond_names,
            ):
                bond_idcs = set(bond_idcs)
                if bond_idcs in already_computed_bond:
                    continue
                already_computed_bond.append(bond_idcs)
                bond_values.append(bond_value)
                bond_names.append(bond_name)
            for angle_idcs, angle_value, angle_name in zip(
                monomer.orig_angle_idcs,
                monomer.angle_values.T,
                monomer.angle_names,
            ):
                angle_idcs = set(angle_idcs)
                if angle_idcs in already_computed_angle:
                    continue
                already_computed_angle.append(angle_idcs)
                angle_values.append(angle_value)
                angle_names.append(angle_name)
            for dihedral_idcs, dihedral_value, dihedral_name in zip(
                monomer.orig_dihedral_idcs,
                monomer.dihedral_values.T,
                monomer.dihedral_names,
            ):
                dihedral_idcs = set(dihedral_idcs)
                if dihedral_idcs in already_computed_dihedral:
                    continue
                already_computed_dihedral.append(dihedral_idcs)
                dihedral_values.append(dihedral_value)
                dihedral_names.append(dihedral_name)
        self.bond_values = np.stack(bond_values, axis=1)
        self.bond_names = np.stack(bond_names, axis=0)
        try:
            self.angle_values = np.stack(angle_values, axis=1)
            self.angle_names = np.stack(angle_names, axis=0)
        except ValueError:
            self.angle_values = np.empty((len(self.bond_values), 0), dtype=np.float32)
            self.angle_names = np.empty((0,), dtype=np.dtype('<U1'))
        try:
            self.dihedral_values = np.stack(dihedral_values, axis=1)
            self.dihedral_names = np.stack(dihedral_names, axis=0)
        except ValueError:
            self.dihedral_values = np.empty((len(self.bond_values), 0), dtype=np.float32)
            self.dihedral_names = np.empty((0,), dtype=np.dtype('<U1'))
    
    @property
    def descriptor_values(self) -> np.ndarray:
        if self.bond_values is None:
            raise Exception("Compute descriptors first!")
        return np.concatenate([self.bond_values, self.angle_values, self.dihedral_values], axis=1)

    @property
    def descriptor_names(self) -> np.ndarray:
        if self.bond_values is None:
            raise Exception("Compute descriptors first!")
        return np.concatenate([self.bond_names, self.angle_names, self.dihedral_names])
    
    def sample_uniform(self, n_samples):
        descriptor_values = self.descriptor_values
        n_frames, n_descriptors = descriptor_values.shape
        samples_per_descriptor = max(n_samples // n_descriptors, 1)

        descriptors_pool = descriptor_values.T.copy()
        if n_samples > descriptors_pool.shape[-1]:
            raise Exception(f'Number of samples ({n_samples}) > Number of dataset frames ({descriptors_pool.shape[-1]})')
        sampled_idcs = np.array([], dtype=int)
        while len(sampled_idcs) < n_samples:
            sweep_sampled_idcs = []
            for descriptor in descriptors_pool:
                sampling_step = max(1, len(descriptor) // samples_per_descriptor)
                sweep_sampled_idcs.append(np.argsort(descriptor)[::sampling_step])
            sweep_sampled_idcs = np.unique(np.concatenate(sweep_sampled_idcs))
            sampled_idcs = np.union1d(sampled_idcs, sweep_sampled_idcs)
            not_sampled_idcs = np.setdiff1d(np.arange(n_frames), sampled_idcs)
            descriptors_pool = descriptor_values.copy()[not_sampled_idcs].T
        return sampled_idcs[:n_samples]