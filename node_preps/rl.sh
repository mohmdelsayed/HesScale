retries=1
until pip install --no-index node_preps/mujoco_py-2.1.2.14-py3-none-any.whl || ((retries++ >= max_retries))
do
    echo pip install mujoco-py failed, trying again in 1 minute
    sleep 60
done

retries=1
until pip install --no-index --find-links node_preps/ node_preps/tyro-0.6.0-py3-none-any.whl || ((retries++ >= max_retries))
do
    echo pip install tyro failed, trying again in 1 minute
    sleep 60
done

