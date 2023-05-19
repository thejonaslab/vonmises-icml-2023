""" Run generate_expectations_pipeline.py to reproduce paper figures. """
import subprocess

output = subprocess.run(f'python generate_expectations_pipeline.py --input_dir results/conformation.dists-nmrshiftdb '
                        f'--working_dir results/conformation.expectations-nmrshiftdb-vm-preds --restrict_to_vm_preds '
                        f'--num_workers 32', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_pipeline.py --input_dir results/conformation.dists-nmrshiftdb '
                        f'--working_dir results/conformation.expectations-nmrshiftdb-non-aromatic-ring '
                        f'--restrict_to_non_aromatic_ring --num_workers 32', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_pipeline.py --input_dir results/conformation.dists-nmrshiftdb '
                        f'--working_dir results/conformation.expectations-nmrshiftdb-all '
                        f'--num_workers 32', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_pipeline.py --input_dir results/conformation.dists-gdb '
                        f'--working_dir results/conformation.expectations-gdb-vm-preds --restrict_to_vm_preds '
                        f'--num_workers 32', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_pipeline.py --input_dir results/conformation.dists-gdb '
                        f'--working_dir results/conformation.expectations-gdb-non-aromatic-ring '
                        f'--restrict_to_non_aromatic_ring --num_workers 32', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_pipeline.py --input_dir results/conformation.dists-gdb '
                        f'--working_dir results/conformation.expectations-gdb-all '
                        f'--num_workers 32', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

