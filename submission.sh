#!/bin/bash
                                       ## REQUIRED: #!/bin/bash must be on the 1st line
	                               ## and it must be the only string on the line
#SBATCH --job-name=cellaap          ## Name of the job for the scheduler
#SBATCH --account=ajitj99             ## Generally your PI's uniqname will go here
#SBATCH --partition=gpu           ## name of the queue to submit the job to.
                                       ## (Choose from: standard, debug, largemem, gpu) 
#SBATCH --gpus=1                 ## if partition=gpu, number of GPUS needed
                                  ## make the directive = #SBATCH, not ##SBATCH 
#SBATCH --nodes=1                      ## number of nodes you are requesting
#SBATCH --ntasks=1                     ## how many task spaces do you want to reserve
##SBATCH --cpus-per-task=1             ## how many cores do you want to use per task
#SBATCH --time=0-08:00:00                 ## Maximum length of time you are reserving the 
	                              ## resources for (bill is based on time used)
#SBATCH --mem-per-gpu=10000m
##SBATCH --mem=5g                       ## Memory requested per core
#SBATCH --mail-user=ajitj@umich.edu  ## send email notifications to umich email listed
#SBATCH --mail-type=BEGIN,END                ## when to send email (standard values are:
                                       ## NONE,BEGIN,END,FAIL,REQUEUE,ALL.
I=$SLURM_ARRAY_TASK_ID
#SBATCH -o $I.log

python batch_inference.py

