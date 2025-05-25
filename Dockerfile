FROM rockylinux:9.3.20231119-minimal

# Install Python
RUN microdnf -y update
RUN microdnf -y install python3

# Set the working directory
WORKDIR /app

# Copy the requirements file (if you have one)
COPY requirements.txt ./

# Install dependencies (if you have any)
RUN microdnf -y install python3-pip
RUN pip3 install -r requirements.txt

# Copy the rest of your application code
COPY images images
COPY waveletTk.py .

# Specify the command to run when the container starts
#CMD ["python3", "waveletTk.py"] 