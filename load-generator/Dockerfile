# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script and HTML file
COPY load_generator.py .
COPY index.html .

# Make port 5002 available to the world outside this container
EXPOSE 5002

# Define environment variable (optional, can be useful)
ENV FLASK_APP=load_generator.py
# ENV FLASK_RUN_HOST=0.0.0.0 # Already set in app.run()

# Run load_generator.py when the container launches
# Use gunicorn for a more production-ready server (optional)
# CMD ["gunicorn", "--bind", "0.0.0.0:5002", "load_generator:app"]
# Or simply run the Flask development server (as used in the script)
CMD ["python", "load_generator.py"] 