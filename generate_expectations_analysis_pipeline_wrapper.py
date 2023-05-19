""" Run generate_expectations_analysis_pipeline.py to reproduce paper figures. """
import subprocess

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-nmrshiftdb-vm-preds '
                        f'--working_dir results/conformation.expectations-analysis-nmrshiftdb-vm-preds '
                        f'--num_workers 8', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-nmrshiftdb-non-aromatic-ring '
                        f'--working_dir results/conformation.expectations-analysis-nmrshiftdb-non-aromatic-ring '
                        f'--num_workers 8', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-nmrshiftdb-all '
                        f'--working_dir results/conformation.expectations-analysis-nmrshiftdb-all '
                        f'--num_workers 8', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-gdb-vm-preds '
                        f'--working_dir results/conformation.expectations-analysis-gdb-vm-preds '
                        f'--max_dist_to_analyze 8 --num_workers 8', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-gdb-non-aromatic-ring '
                        f'--working_dir results/conformation.expectations-analysis-gdb-non-aromatic-ring '
                        f'--num_workers 8', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-gdb-all '
                        f'--working_dir results/conformation.expectations-analysis-gdb-all '
                        f'--num_workers 8', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-nmrshiftdb-vm-preds '
                        f'--working_dir results/conformation.expectations-analysis-nmrshiftdb-vm-preds-ex-torsional '
                        f'--num_workers 8 --ignore_torsional_diffusion', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-nmrshiftdb-non-aromatic-ring '
                        f'--working_dir results/conformation.expectations-analysis-nmrshiftdb-non-aromatic-ring-ex-torsional '
                        f'--num_workers 8 --ignore_torsional_diffusion', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-nmrshiftdb-all '
                        f'--working_dir results/conformation.expectations-analysis-nmrshiftdb-all-ex-torsional '
                        f'--num_workers 8 --ignore_torsional_diffusion', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-gdb-vm-preds '
                        f'--working_dir results/conformation.expectations-analysis-gdb-vm-preds-ex-torsional '
                        f'--num_workers 8 --ignore_torsional_diffusion', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-gdb-non-aromatic-ring '
                        f'--working_dir results/conformation.expectations-analysis-gdb-non-aromatic-ring-ex-torsional '
                        f'--num_workers 8 --ignore_torsional_diffusion', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')

output = subprocess.run(f'python generate_expectations_analysis_pipeline.py '
                        f'--input_dir results/conformation.expectations-gdb-all '
                        f'--working_dir results/conformation.expectations-analysis-gdb-all-ex-torsional '
                        f'--num_workers 8 --ignore_torsional_diffusion', shell=True, capture_output=True)
print('Subprocess: ')
print(f'args: {output.args}')
print(f'return code: {output.returncode}')
print(f'output: {output.stdout.decode("utf-8")}')
print(f'error: {output.stderr.decode("utf-8")}\n')
