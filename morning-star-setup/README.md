# Running on Morning Star

1. Upload the diffvg-and-setup.zip file to Morning Star.
2. Open a terminal tab on Morning Star and unzip the file (unzip diffvg-and-setup.zip)
3. chmod +x setup.sh
4. ./setup.sh

Be patient! Sometimes it looks like pip is doing nothing, but it will eventually complete the installation.

5. Zip the project folder (preferably without the .venv and .git folders), upload to Morning Star then open the notebook selecting myenv as kernel. If it doesn't appear, reload the page.


You can also run the commands individually instead of using the setup script:
```bash
# create virtual environment
virtualenv venv
source venv/bin/activate
pip config set global.index-url https://artifacts.uottawa.ca/artifactory/api/pypi/python/simple

# diffvg
pip install --ignore-installed torch torchvision numba scikit-image svgwrite svgpathtools cssutils torch-tools visdom wheel cmake ffmpeg setuptools
pip install -v --no-build-isolation ./diffvg-patched

# project dependencies
pip install --ignore-installed keras ipykernel shapely

# register kernel
python -m ipykernel install --user --name=myenv

# then select myenv as kernel when opening a notebook
```