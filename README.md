# VonMisesNet: Von Mises Mixture Distributions for Molecular Conformation Generation 

This repository implements Von Mises Mixture Distributions for Molecular Conformation Generation. VonMisesNet is a new 
graph neural network that can generate conformations for molecules in a way that is both physically accurate with 
respect to the Boltzmann distribution and orders of magnitude faster than existing sampling methods.  


## Installation
conformation-vonmises can be installed from source:
1. ```git clone --branch ks8/release https://github.com/thejonaslab/conformation-vonmises.git```
2. ```cd conformation-vonmises```
3. ```conda env create -n $NAME -f environment.yml```, where $NAME is the desired name of the environment
4. ```conda activate $NAME```
5. ```pip install -e .```

## Documentation
Documentation can be found at ```docs/_build/html/vonmises.html``` (will go on ReadTheDocs for release).

## Conformation Generation
VonMisesNet can be used to generate conformations for arbitrary molecules. To do so, create a csv file containing a 
SMILES string on each line, with no header. Then, run:

```
python generate_confs_from_smiles.py --csv_path <path-to-smiles-csv> --out_path <path-to-output-pickle> --model_path <path-to-model-params> --model_config_path <path-to-model-config> --num_confs <#-confs-to-generate> --filter_confs
```

- ```--csv_path```: Path to the CSV containing SMILES strings.
- ```--out_path```: Path to the output pickle file, which contains RDKit mol objects with generated conformations and prediction metadata. 
- ```--model_path```: Path to the VonMisesNet model parameters. 
- ```--model_config_path```: Path to the VonMisesNet model config yaml file.
- ```--num_confs```: Number of conformations to generate.
- ```--filter_confs```: Optional filtering of conformations which violate minimum atomic distance thresholds.

For the model trained on NMRShiftDB data described in the paper, use:

```--model_path models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.best.chk```

```--model_config_path models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.yaml``` 


For the model trained on GDB-17 data described in the paper, use:

```--model_path models/gdb_20220924_fix_chirality.gdb_20220924.644918830879.best.chk``` 

```--model_config_path models/gdb_20220924_fix_chirality.gdb_20220924.644918830879.yaml```

See ```generate_confs_from_smiles.py``` for more options. Here is an example using the provided ```example_smiles.csv``` 
script, which contains ethane and ibuprofen:

```
python generate_confs_from_smiles.py --csv_path example_smiles.csv --out_path example_smiles.pkl --model_path models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.best.chk --model_config_path models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.yaml --num_confs 560 --filter_confs
```

This script utilizes a ```Predictor``` class and a ```generate_confs``` function contained in ```vonmises.predictor```. 
There are multiple additional options available when using these directly, such as making predictions for a list of 
RDKit molecule objects instead of SMILES strings. See the documentation for more details. Here is an example where we 
input an ethane molecule that has an initial 3D geometry:

```
from rdkit import Chem
from rdkit.Chem import AllChem
from vonmises.predictor import Predictor, generate_confs

mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
AllChem.EmbedMolecule(mol)

predictor = Predictor(model_path="models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.best.chk", model_config_path="models/nmrshiftdb_20220924_fix_chirality.nmrshiftdb_20220924.644553492873.yaml", use_cuda=False)
preds, metas = predictor.predict([mol])

new_mol = generate_confs(predictions=preds[0], num_confs=560, filter_confs=True)
```

## Training Data
There are currently two datasets available to train VonMisesNet: ```nmrshiftdb-pt-conf-mols.db```, which contains 
molecules from NMRShiftDB, and ```GDB-17-stereo-pt-conf-mols.db```, which contains molecules from GDB-17. 

To use these for training:
1. ```cd conformation-vonmises```
2. ```cp /jonaslab/projects/conformation/data/mol-data/data.targ.gz .``` (This will be moved to file server for actual release.) 
3. ```tar -xvf data.tar.gz```
4. Run ```python generate_db_targets.py```, which extracts training targets from the molecules in these datasets.

To create your own dataset for training:
1. Store generated conformations in RDKit molecule objects.
2. Place these objects in a dictionary, mapping from unique molecule ID integer to a binary representation of the 
molecule object. This can be achieved via the ```ToBinary()``` method in RDKit. 
3. Save this dictionary as a pickle file.
4. Run ```python mols_to_db.py <path-to-pickle> <output-db-name>```, where ```<path-to-pickle>``` is the local path to 
the pickle file and ```<output-db-name>``` is a prefix for the database output filename, which will be saved in the 
```data/mol-data``` directory.
5. Open ```generate_db_targets.py``` and add an entry to the TARGET dictionary. The key should be a unique convenient 
name, and the value should be a dictionary, where ```"data"``` is the path to the database file from the previous step 
and ```"targets"``` should be set to ```DEFAULT_TARGET_KINDS```.
6. Run ```python generate_db_targets.py```

## Training
To train VonMisesNet, run ```python train.py expconfig/<config-yaml> <convenient-experiment-name>```. 

```<config-yaml>``` is a yaml configuration file that specifies metadata, data loader, and neural network parameters 
for training. The ```dataset``` parameter ```mol_db``` points to the molecule dataset database file, and the 
```dataset ``` parameter ```target_file``` is the key that was used for that dataset in ```generate_db_targets.py```. 
```<convenient-experiment-name>``` is a name that will be used in checkpoint files (saved in a directory named 
```checkpoints```) and tensorboard logs (saved in a directory named ```tblogs```).  

To reproduce the models in the paper, use the ```expconfig/nmrshiftdb_best.yaml``` and ```expconfig/gdb17_best.yaml``` 
configuration files for the NMRShiftDB data and the GDB-17 data, respectively.

## Reproducing Paper Evaluations 
To reproduce figures in the paper:

1. ```cd conformation-vonmises```
2. ```cp /jonaslab/projects/conformation/data/mol-data/results.tar.gz .``` (This will be moved to file server for actual release.) 
3. ```tar -xvf results.tar.gz```
4. Follow the step-by-step instructions in ```paper_figures.ipynb```.