#!/bin/bash
#SBATCH --job-name=pretrain                                 # Job name
#SBATCH --cpus-per-task=2                                   # Ask for 2 CPU
#SBATCH --gres=gpu:1                                        # Ask for 1 GPU
#SBATCH --mem=16000M                                        # Ask for 16 GB of RAM
#SBATCH --time=0-05:30:00                                   # Duration: max 5.5 hour
#SBATCH --output=/scratch/simon88812/logs/slurm-%j-%x.out   # log file
#SBATCH --error=/scratch/simon88812/logs/slurm-%j-%x.error  # log file

# Arguments
# $0: Path to code directory. Directory must be already present on the remote machine.
# Copy code dir to the compute node and cd there
INIT_DIR=$(pwd)
echo "$INIT_DIR"
rsync -av --relative ./stability/ $SLURM_TMPDIR --exclude ".git" --exclude "venv" # copies the _content_ of stability directory to the slurm tmpdir
cd $SLURM_TMPDIR
ls
echo "Loading modules"
module load gcc/9.3.0 opencv python scipy-stack
module load python/3.10 cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install numpy scikit-learn scipy opencv --no-index
pip install torch torchvision --no-index

echo "Currently using:"
echo $(which python)
echo "in:"
echo $(pwd)
ls

# python download.py
cd $SLURM_TMPDIR/stability
ls
echo "Launching training"
python -m train
echo "Training ended"

rsync -av $SLURM_TMPDIR/stability/checkpoints/ $INIT_DIR/stability/checkpoints/
