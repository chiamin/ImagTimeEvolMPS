import sys, os

def write_para (jobname, partition="", cpus=1, mem='1GB', hours=168, constraint='', f=sys.stdout):
    print('#!/bin/bash',file=f)
    print('#SBATCH --job-name='+jobname,file=f)
    print('#SBATCH --partition='+partition,file=f)
    print('#SBATCH --cpus-per-task='+str(cpus),file=f)
    print('#SBATCH --mem='+mem,file=f)
    print('#SBATCH --time='+str(hours)+':00:00',file=f)
    print('#SBATCH --constraint='+constraint,file=f)
    print('',file=f)

def write_script (exe, input_file='', output_file='', f=sys.stdout):
    # Go to the working directory
    workdir = os.getcwd()
    print('cd '+workdir,file=f)
    # Print information
    print('echo "Working directory:  '+workdir+'"',file=f)

    # The output file should be in the submitted directory
    if output_file != '':
        output = os.getcwd()+'/'+output_file
        # Write input file to the output file
        print('cat '+input_file+' >> '+output,file=f)
        # Run
        print('srun '+exe+' '+input_file+' >> '+output,file=f)
    else:
        print('srun '+exe+' '+input_file,file=f)

def write_info (f=sys.stdout):
    print('echo "Node:               $SLURM_JOB_NODELIST"',file=f)
    print('echo "Slurm User:         $SLURM_JOB_USER"',file=f)
    print('echo "Job ID:             $SLURM_JOB_ID"',file=f)
    print('echo "Job Name:           $SLURM_JOB_NAME"',file=f)
    print('echo "Partition:          $SLURM_JOB_PARTITION"',file=f)
    print('echo "Number of nodes:    $SLURM_JOB_NUM_NODES"',file=f)
    print('echo "Number of cups:     $SLURM_CPUS_PER_TASK"',file=f)
    print('echo "Memory:             $SLURM_MEM"',file=f)
    print('echo "Time limit:         $SBATCH_TIMELIMIT"',file=f)
    print('echo "Constrain:          $SBATCH_CONSTRAINT"',file=f)
    print('echo "Submitted From:     $SLURM_SUBMIT_DIR"',file=f)
    print('',file=f)

if __name__ == '__main__':
    write_para (jobname='test', partition='chiaminq', cpus=20, mem='3GB', hours=24)
    print()
    write_info ()
    print()
    write_script ('echo test', 'input', 'output')
