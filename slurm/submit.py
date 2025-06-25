import myslurm as slurm
import os, sys

if __name__ == '__main__':
    para = 'para'
    exe = 'julia ImagMPSComb.jl '+para+' '

    if os.path.isdir(data_dir):
        ans = input(data_dir+" already exist. Is it OK?")
        if ans not in ['y','Y']:
            exit()

    for nsteps in [20,40,60,80,100,120,140,160,180,200]:
        jobname = 'ntau'+str(nsteps)
        with open('slurm_'+jobname,'w') as f:
            # Set the resources you need
            slurm.write_para (jobname=jobname, f=f)
            # Write information. Don't chanage.
            slurm.write_info (f=f)

            #slurm.write_script (exe='./quench.exe', input_file='input', output_file='output', f=f)
            exe = exe + ' --nsteps='+str(nsteps)+' --suffix='+jobname+' '
            slurm.write_script (exe=exe, f=f)

    #os.system ('sbatch '+slurm_script)
