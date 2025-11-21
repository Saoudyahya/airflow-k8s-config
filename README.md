helm install airflow apache-airflow/airflow --namespace airflow --create-namespace --set postgresql.image.tag="latest"
