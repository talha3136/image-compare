# echo "BUILD START"
# pip install -r requirements.txt
# python3.12 manage.py collectstatic --noinput --clear
# echo "BUILD END"

#!/bin/bash

echo "BUILD START"

# Install CMake (required for building dlib from source)
apt-get update && apt-get install -y cmake

# Install required Python packages
pip install -r requirements.txt

# Collect static files to the correct directory
python3.12 manage.py collectstatic --noinput --clear

echo "BUILD END"
