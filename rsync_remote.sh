echo "Overwrite local with remote"
sleep 2
rsync -av simon88812@ift6759.calculquebec.cloud:~/sbatch/stability . --exclude ".git" --exclude ".venv" --exclude "venv"
