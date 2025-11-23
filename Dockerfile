# Use Airflow 3
FROM apache/airflow:3.1.3-python3.12

COPY requirements.txt .
# Unpinned requirements are good here, let Airflow 3 pick the newest versions
RUN pip install --no-cache-dir -r requirements.txt
