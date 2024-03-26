module load python/3.8.10

# Prepare virtualenv
max_retries=10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

retries=1
until pip install --no-index -r requirements_computecanada.txt || ((retries++ >= max_retries))
do
    echo pip install -r requirements_computecanada.txt failed, trying again in 1 minute
    sleep 60
done

retries=1
until pip install --no-index --find-links node_preps/ node_preps/tikzplotlib-0.10.1-py3-none-any.whl || ((retries++ >= max_retries))
do
    echo pip install tyro failed, trying again in 1 minute
    sleep 60
done

retries=1
until pip install -e ~/projects/def-ashique/farrahi/HesScale/DeepOBS/ || ((retries++ >= max_retries))
do
    echo pip install -e ~/projects/def-ashique/farrahi/HesScale/DeepOBS/ failed, trying again in 1 minute
    sleep 60
done

retries=1
until pip install -e ~/projects/def-ashique/farrahi/HesScale/backpack/ || ((retries++ >= max_retries))
do
    echo pip install -e ~/projects/def-ashique/farrahi/HesScale/backpack/ failed, trying again in 1 minute
    sleep 60
done

retries=1
until pip install -e . || ((retries++ >= max_retries))
do
    echo pip install -e HesScale failed, trying again in 1 minute
    sleep 60
done

