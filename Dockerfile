
# 1. Use an official Python runtime as a parent image
FROM python:3.13-slim

# 2. Set environment variables to prevent Python from writing .pyc files
# and to ensure the output of the container's console is logged immediately
ENV PYTHONUNBUFFERED 1

# 3. Install system dependencies
# We install the necessary packages to build some dependencies (like psycopg2)
RUN apt-get update \
    && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && apt-get clean

# 4. Set the working directory in the container
WORKDIR /app

# 5. Copy the requirements file into the container
# This allows us to install dependencies before copying the whole project
COPY requirements.txt /app/

# 6. Install Python dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the Django project into the container
COPY . /app/

# 8. Expose the port that the app will run on
EXPOSE 8000

# 9. Run Django migrations (optional but useful for setting up databases)
# Make sure your database is configured in settings.py (PostgreSQL)
# RUN python manage.py migrate --noinput

# 10. Use Tini to handle signals gracefully (optional)
# This is useful to handle SIGTERM and SIGINT correctly, especially when running Django in the foreground
RUN apt-get install -y tini

ENTRYPOINT ["/usr/bin/tini", "--"]

# 11. Start the Django development server or use Gunicorn for production
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
