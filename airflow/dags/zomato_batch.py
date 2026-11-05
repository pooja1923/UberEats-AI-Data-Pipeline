from datetime import datetime
from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.standard.operators.bash import BashOperator      # Airflow 3 import

DBT = "/opt/airflow/dbt_venv/bin/dbt"
DBT_PROJECT = "/opt/airflow/dbt/zomato"

COPY_RAW = [
    "USE WAREHOUSE ZOMATO_WH",
    "COPY INTO ZOMATO.RAW.restaurants FROM @Zomato.Raw.ZOMATO_RAW_STAGE/restaurants/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE';",
    "COPY INTO ZOMATO.RAW.USERS FROM @Zomato.Raw.ZOMATO_RAW_STAGE/users/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE';",
    "COPY INTO ZOMATO.RAW.FOOD FROM @Zomato.Raw.ZOMATO_RAW_STAGE/food/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE' ;",
    "COPY INTO ZOMATO.RAW.MENU FROM @Zomato.Raw.ZOMATO_RAW_STAGE/menu/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE' ;",
    "COPY INTO ZOMATO.RAW.ORDERS FROM @Zomato.Raw.ZOMATO_RAW_STAGE/orders/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE';",
    "COPY INTO ZOMATO.RAW.ORDER_ITEMS FROM @Zomato.Raw.ZOMATO_RAW_STAGE/order_items/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE' ;",
    "COPY INTO ZOMATO.RAW.REVIEWS FROM @Zomato.Raw.ZOMATO_RAW_STAGE/reviews/ FILE_FORMAT = (FORMAT_NAME = 'ZOMATO.RAW.CSV_FMT') ON_ERROR = 'CONTINUE';",
]

with DAG(
    dag_id="zomato_batch",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["zomato", "dbt", "snowflake"],
    doc_md=__doc__,
) as dag:

    reload_raw = SQLExecuteQueryOperator(
        task_id="reload_raw", conn_id="snowflake_default",
        sql=COPY_RAW, split_statements=True, autocommit=True,
    )

    dbt_build_core = BashOperator(
        task_id="dbt_build_core",
        bash_command=f"{DBT} build --exclude tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT}",
    )

    enrich_reviews = BashOperator(
        task_id="enrich_reviews",
        bash_command=f"python /opt/airflow/ai/enrich_reviews.py",
    )

    dbt_build_ai = BashOperator(
        task_id = "dbt_build_ai",
        bash_command=f"{DBT} build --select tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT}"
    )

    reload_raw >> dbt_build_core >> enrich_reviews >> dbt_build_ai