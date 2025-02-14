# echo "BUILD START"
# pip install -r requirements.txt
# python3.12 manage.py collectstatic --noinput --clear
# echo "BUILD END"

#!/bin/bash

echo "BUILD START"

# Install required packages
pip install -r requirements.txt

# Collect static files to the correct directory
python3.12 manage.py collectstatic --noinput --clear

echo "BUILD END"
