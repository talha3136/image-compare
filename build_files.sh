# echo "BUILD START"
# pip install -r requirements.txt
# python3.12 manage.py collectstatic --noinput --clear
# echo "BUILD END"

#!/bin/bash


#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Print commands before executing them
set -x

echo "Starting Django build process..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "Running database migrations..."
python manage.py migrate

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Compile messages for internationalization
if [ -d "locale" ]; then
    echo "Compiling translation messages..."
    python manage.py compilemessages
fi

# Run tests
echo "Running tests..."
python manage.py test

# Check for security vulnerabilities in dependencies
echo "Checking for security vulnerabilities..."
pip install safety
safety check

# Generate documentation if Sphinx is configured
if [ -d "docs" ]; then
    echo "Building documentation..."
    cd docs
    make html
    cd ..
fi

# Create a production-ready .env file if it doesn't exist
if [ ! -f ".env.prod" ] && [ -f ".env.example" ]; then
    echo "Creating production environment file from example..."
    cp .env.example .env.prod
    echo "IMPORTANT: Update .env.prod with production values!"
fi

echo "Build process completed successfully!"

# Deactivate virtual environment
deactivate


# echo "BUILD START"

# # Install CMake (required for building dlib from source)
# apt-get update && apt-get install -y cmake build-essential

# # Install required Python packages
# pip install -r requirements.txt

# # Collect static files to the correct directory
# python3.12 manage.py collectstatic --noinput --clear

# echo "BUILD END"
