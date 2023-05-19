""" PyTorch Dataset. """
import numpy as np
import os
from typing import Dict, List

from rdkit import Chem
from sqlalchemy import event, create_engine, MetaData, select, Table, text
from sqlalchemy.engine import Engine
from torch.utils.data import Dataset, Sampler

from vonmises.featurize_mols import FeaturizeMol
from vonmises.graph_builders import BuildVonMisesGraph


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Try and increase performance: https://blog.devart.com/increasing-sqlite-performance.html.

    :param dbapi_connection: Database connection.
    :param connection_record: Connection record.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA locking_mode=EXCLUSIVE")
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.execute("PRAGMA query_only=TRUE")
    # If the argument N is positive then the suggested cache size is set to N.
    # If the argument N is negative, then the number of cache pages is adjusted to be a number
    # of pages that would use approximately abs(N*1024) bytes of memory based on the current page size.
    cursor.execute("PRAGMA cache_size=-5000000")
    cursor.close()

    
class TargetsLoader:
    """
    Database connection for loading and deserializing targets from a database.
    """
    def __init__(self, db_filename, row_deserialize, table_name):
        """
        :param db_filename: Filename of database.
        :param row_deserialize: Function to use for row deserialization.
        :param table_name: Table name.
        """
        assert os.path.exists(db_filename)
        self.db_filename = db_filename
        self.row_deserialize = row_deserialize
        self.engine = create_engine(f"sqlite+pysqlite:///{db_filename}?immutable=1", future=True)
        self.conn = self.engine.connect()
        self.table_name = table_name

    def __getitem__(self, mol_id):
        """
        :param mol_id: Molecule ID.
        :return: Deserialized row.
        """
        row = self.conn.execute(text(f'select * from {self.table_name} where mol_id={mol_id}')).one()
        return self.row_deserialize(row)


def db_row_to_chirality_targets(row) -> Dict[str, np.ndarray]:
    """
    Deserialize a row from a chirality targets database.

    :param row: Database row.
    :return: Deserialized chirality targets.
    """
    atom_n = row['atom_n']
    chiralities = np.frombuffer(row['chiralities_bin'], dtype=np.float16)
    if chiralities.size != 0:
        chiralities = chiralities.reshape(atom_n, -1)
    atom_indices = np.frombuffer(row['atom_idx_bin'],
                                 dtype=np.int32)

    return {'chiralities': chiralities,
            'atom_indices': atom_indices}


def db_row_to_length_targets(row) -> Dict[str, np.ndarray]:
    """
    Deserialize a row from a bond length targets database.

    :param row: Database row.
    :return: Deserialized bond length targets.
    """
    bond_n = row['bond_n']
    bond_lengths = np.frombuffer(row['bond_lengths_bin'],
                                  dtype=np.float16).reshape(bond_n, -1)
    bond_indices = np.frombuffer(row['bond_idx_bin'],
                                 dtype=np.int32)
    
    return {'lengths': bond_lengths,
            'bond_indices': bond_indices}


def db_row_to_angle_targets(row) -> Dict[str, np.ndarray]:
    """
    Deserialize a row from a bond angle targets database.

    :param row: Database row.
    :return: Deserialized bond angle targets.
    """
    angle_n = row['angle_n']
    angles = np.frombuffer(row['angles_bin'],
                            dtype=np.float16).reshape(angle_n, -1)
    atom_indices = np.frombuffer(row['atom_idx_bin'],
                                 dtype=np.int32).reshape(angle_n, 3)
    
    return {'angles': angles,
            'angle_triplets': atom_indices}


def db_row_to_torsion_targets(row) -> Dict[str, np.ndarray]:
    """
    Deserialize a row from a torsion targets database.

    :param row: Database row.
    :return: Deserialized torsion targets.
    """
    bond_n = row['bond_n']
    if bond_n == 0:
        return {'angles': np.empty([0, 0]),
                'bond_indices': np.empty([0]),
                'bond_classes': np.empty([0]),
                'atom_index_tuples': np.empty([0, 0]),
                'bond_indices_with_chirality': np.empty([0])}

    angles = np.frombuffer(row['angles_bin'],
                            dtype=np.float16).reshape(bond_n, -1)
    assert angles.shape
    bond_indices = np.frombuffer(row['bond_indices_bin'],
                                 dtype=np.int32)
    atom_indices = np.frombuffer(row['atom_indices_bin'],
                                 dtype=np.int32).reshape(bond_n, 4)
    bond_classes = np.frombuffer(row['bond_classes_bin'],
                                 dtype=np.int32)
    bond_indices_with_chirality = np.frombuffer(row['bond_indices_with_chirality_bin'],
                                                dtype=np.int32)
    
    return {'angles': angles,
            'bond_indices': bond_indices,
            'bond_classes': bond_classes,
            'atom_index_tuples': atom_indices,
            'bond_indices_with_chirality': bond_indices_with_chirality}


class GraphDBDataset(Dataset):
    """
    Dataset class for loading molecular graphs and targets, using on-disk sqlite databases.

    Note: we delay connecting to the database until we request the first item _and_ we check to see if the PID is the
    same; this is to deal with Pytorch's "numworkers" forking and difficulties with Python Multiprocessing.
    """
    def __init__(self, db_path: str,
                 molecule_ids: List[int],
                 torsion_targets_db: str,
                 angle_targets_db: str,
                 len_targets_db: str,
                 chirality_targets_db: str,
                 cache_features: bool = False, atom_feat_config: Dict = None,
                 bond_feat_config: Dict = None, preprocess_config: Dict = None,
                 graph_build_config: Dict = None):
        """
        :param db_path: Path to SQLite database containing molecules.
        :param molecule_ids: Molecule ids in SQLite database.
        :param torsion_targets_db: Path to torsion targets database.
        :param angle_targets_db: Path to bond angle targets database.
        :param len_targets_db: Path to bond length targets database.
        :param chirality_targets_db: Path to chirality targets database.
        :param cache_features: Whether or not to cache features in memory for faster loading.
        :param atom_feat_config: Dictionary of atom parameters.
        :param bond_feat_config: Dictionary of bond parameters.
        :param graph_build_config: Dictionary of graph building parameters.
        """
        super(Dataset, self).__init__()
        if graph_build_config is None:
            graph_build_config = {}
        if bond_feat_config is None:
            bond_feat_config = {}
        if atom_feat_config is None:
            atom_feat_config = {}
        self.featurize_mol = FeaturizeMol(atom_feat_config, bond_feat_config, preprocess_config)
        self.build_graph = BuildVonMisesGraph(**graph_build_config)
        self.molecule_ids = molecule_ids
        self.db_path = db_path
        self.torsion_targets_db = torsion_targets_db
        self.angle_targets_db = angle_targets_db
        self.len_targets_db = len_targets_db
        self.chirality_targets_db = chirality_targets_db
        self.sql_engine = None
        self.cache_features = cache_features
        self.cache = dict()
        self.created_pid = os.getpid()

    def _connect_dbs(self):
        """
        Function to be run after forking to make sure we connect the databases; always reconnect on process change.
        """
        self.sql_engine = create_engine(f"sqlite+pysqlite:///{self.db_path}?immutable=1", future=True)
        self.molecules_table = Table("molecules", MetaData(), autoload_with=self.sql_engine)
        self.molecules_conn = self.sql_engine.connect()
        self.torsion_targets_loader = TargetsLoader(self.torsion_targets_db, db_row_to_torsion_targets, 'torsions')
        self.angle_targets_loader = TargetsLoader(self.angle_targets_db, db_row_to_angle_targets, 'angles')
        self.length_targets_loader = TargetsLoader(self.len_targets_db, db_row_to_length_targets, 'lengths')
        self.chirality_targets_loader = TargetsLoader(self.chirality_targets_db, db_row_to_chirality_targets,
                                                      'chiralities')

    def __len__(self) -> int:
        return len(self.molecule_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return the graph representation of a molecule.

        :param idx: Which molecule to process.
        :return: Graph dictionary.
        """
        reconnect_db = False
        my_pid = os.getpid()
        if self.created_pid != my_pid:
            print(f"dataloader running in new process, pids: {self.created_pid} -> {my_pid}; reconnecting DB")
            self.created_pid = my_pid
            reconnect_db = True
        elif self.sql_engine is None:
            # numworkers = 0; first time running
            reconnect_db = True
            
        if reconnect_db:
            self._connect_dbs()

        if self.cache_features:
            if idx in self.cache:
                return self.cache[idx]

        # Load molecule
        query = select([self.molecules_table.c.mol]).where(self.molecules_table.c.id == self.molecule_ids[idx])
        res = self.molecules_conn.execute(query).fetchone()
        mol = Chem.Mol(res['mol'])
        uid = self.molecule_ids[idx]

        # Load targets
        chirality_targets = self.chirality_targets_loader[uid]
        length_targets = self.length_targets_loader[uid]
        angle_targets = self.angle_targets_loader[uid]
        torsion_targets = self.torsion_targets_loader[uid]

        # Featurize molecule
        features = self.featurize_mol(mol, angle_targets['angle_triplets'])

        # Build graph representation
        graph = self.build_graph(mol, torsion_targets, angle_targets, length_targets, chirality_targets, features)
        graph['uid'] = np.array([uid])

        # Save to cache
        if self.cache_features:
            self.cache[idx] = graph

        return graph

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))


def create_dataset(mol_db_filename, target_basename, dataset_ids, **exp_config_params) -> GraphDBDataset:
    """
    Create a GraphDBDataset object.

    :param mol_db_filename: Molecule database filename.
    :param target_basename: Target databases base filename.
    :param dataset_ids: List of ids.
    :param exp_config_params: Experiment config parameters.
    :return: GraphDBDataset object.
    """
    return GraphDBDataset(mol_db_filename,
                          dataset_ids,
                          target_basename + '.torsions.db', 
                          target_basename + '.angles.db', 
                          target_basename + '.lengths.db',
                          target_basename + '.chiralities.db',
                          **exp_config_params)


class SubsetSampler(Sampler):
    """
    A dataset sampler for sampling a subset of data for each epoch, in order to have epochs of the same size to
    compare training performance across different datasets.
    """
    def __init__(self, epoch_size, ds_size, shuffle=False, world_size=1, rank=0, seed=None, logging_name=None):
        """
        :param epoch_size: Size of the epoch.
        :param ds_size: Total size of the dataset.
        :param shuffle: Whether or not to shuffle.
        :param world_size: World size.
        :param rank: Rank.
        :param seed: Random seed.
        :param logging_name: Logger name.
        """
        self.epoch_size = epoch_size
        self.ds_size = ds_size
        self.pos = 0
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        if seed is None:
            seed = rank
        self.bit_generator = np.random.PCG64(seed)
        self.rng = np.random.default_rng(self.bit_generator)
        self.compute_idx()
        self.logging_name = logging_name
        if logging_name is not None:
            self.logging_fid = open(f"{logging_name}.samples", 'a')

    def get_state(self):
        """
        Returns the state to reinitialize the sampler.
        """
        return {'pos': self.pos,
                'epoch_size': self.epoch_size,
                'idx': self.idx,
                'ds_size': self.ds_size,
                'bg_state': self.bit_generator.state}

    def set_state(self, state_dict):
        """
        Set the state of the sampler.

        :param state_dict: State dictionary.
        """
        assert self.epoch_size == state_dict['epoch_size']
        assert self.ds_size == state_dict['ds_size']

        self.pos = state_dict['pos']
        self.bit_generator.state = state_dict['bg_state']
        self.idx = state_dict['idx']

    def compute_idx(self):
        """
        Compute index.
        """
        if self.shuffle:
            self.idx = self.rng.permutation(self.ds_size)
        else:
            self.idx = np.arange(self.ds_size)

        self.idx = self.idx[self.idx % self.world_size == self.rank]

    def __len__(self):
        return self.epoch_size // self.world_size

    def __iter__(self):
        for i in range(self.__len__()):
            y = self.idx[self.pos]
            if self.logging_name is not None:
                self.logging_fid.write(f"{y}\n")
                self.logging_fid.flush()
            yield y
            self.pos = (self.pos + 1) % len(self.idx)

            if self.pos == 0:
                self.compute_idx()
