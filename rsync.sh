echo "Sync cluster's stability dir"
rsync -av . simon88812@ift6759.calculquebec.cloud:~/sbatch/stability --exclude ".git" --exclude ".venv" --exclude "venv"
