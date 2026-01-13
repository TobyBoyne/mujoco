https://mujoco.readthedocs.io/en/latest/programming/#building-from-source
https://mujoco.readthedocs.io/en/latest/python.html#building-from-source

### Build mujoco from source
```bash
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build .
cmake --install .
```


### Build python bindings
```bash
cd ../python
uv venv --python 3.12 /tmp/mujoco
source /tmp/mujoco/bin/activate
python -m ensurepip --upgrade
bash make_sdist.sh
```