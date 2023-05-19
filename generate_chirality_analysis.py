from vonmises import mol_utils
from vonmises.sqlalchemy_model import Molecule, create_session
from sqlalchemy import select
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem.Lipinski import RotatableBondSmarts

db_session = create_session("/jonaslab/projects/conformation/data/mol-data/nmrshiftdb-pt-conf-mols.db")
all_molids = [a[0] for a in db_session.execute(select([Molecule.id])).all()]
num_rot_bond_chirality = 0
num_rot_bond_two_chirality = 0
num_mol_rot_bond_chirality = 0
num_mol_rot_bond_chirality_two = 0
num_rot_bonds = 0
for mol_id in tqdm(all_molids):
    mol_has_chirality = False
    mol_has_chirality_two = False
    stmt = select([Molecule]).where(Molecule.id == int(mol_id))
    db_mol = db_session.execute(stmt).one()[0]
    mol = Chem.Mol(db_mol.mol)
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    for bond in rotatable_bonds:
        num_rot_bonds += 1
        chir_result_1 = False
        if len(mol.GetAtomWithIdx(bond[0]).GetNeighbors()) == 3 and mol_utils.count_lone_pairs(mol, bond[0]) == 1:
            chir_result_1 = mol_utils.compute_chirality_all_confs(mol, bond[0])
            chir_result_1 = 0 < sum([x for x in chir_result_1 if x > 0]) / len(chir_result_1) < 1
        chir_result_2 = False
        if len(mol.GetAtomWithIdx(bond[1]).GetNeighbors()) == 3 and mol_utils.count_lone_pairs(mol, bond[1]) == 1:
            chir_result_2 = mol_utils.compute_chirality_all_confs(mol, bond[1])
            chir_result_2 = 0 < sum([x for x in chir_result_2 if x > 0]) / len(chir_result_2) < 1
        if chir_result_1 or chir_result_2:
            num_rot_bond_chirality += 1
            mol_has_chirality = True
        if chir_result_1 and chir_result_2:
            num_rot_bond_two_chirality += 1
            mol_has_chirality_two = True

    if mol_has_chirality:
        num_mol_rot_bond_chirality += 1

    if mol_has_chirality_two:
        num_mol_rot_bond_chirality_two += 1

print(num_rot_bond_chirality, num_rot_bond_two_chirality, num_mol_rot_bond_chirality, num_mol_rot_bond_chirality_two,
      len(all_molids), num_rot_bonds)