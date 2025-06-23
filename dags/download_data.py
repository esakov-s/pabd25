import datetime
from pathlib import Path
import pendulum
import os
import glob

from airflow.sdk import dag, task
import subprocess
import sys



def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


install_package("cianparser")


import pandas as pd

import cianparser


def request_to_cian(n_rooms=1):
    """
    Parse data to data/raw
    :param int n_rooms: The number of flats rooms
    :return None
    """

    moscow_parser = cianparser.CianParser(location="Москва")

    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 5,
            "object_type": "secondary",
        },
    )
    df = pd.DataFrame(data)[["url", "total_meters", "floor", "floors_count", "rooms_count", "price"]]
    return df


@dag(
    dag_id="process_data",
    schedule="0 0 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
)
def ProcessFlats():
    @task
    def get_data(n_rooms):
        data = request_to_cian(n_rooms)
        root = Path("/opt/airflow/dags/files/")
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / f"downloaded_flats_{n_rooms}.csv"
        data.to_csv(data_path, index=False)

    @task
    def preprocess_data():
        """
        Filter, sort and remove duplicates
        """
        root = Path("/opt/airflow/dags/files/")
        file_list = os.listdir(root)[:-1]  # exclude last file with processed data

        df = pd.read_csv(root / file_list[0])
        for i in range(1, len(file_list)):
            data = pd.read_csv(root / file_list[i])
            df_i = pd.DataFrame(data)
            df = pd.concat([df, df_i], axis=0)

        df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
        df = (
            df[["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]]
            .set_index("url_id")
            .sort_index()
        )

        df.drop_duplicates(inplace=True)
        df = df[df["price"] < 100_000_000]
        df = df[df["total_meters"] < 100]

        df["rooms_1"] = df["rooms_count"] == 1
        df["rooms_2"] = df["rooms_count"] == 2
        df["rooms_3"] = df["rooms_count"] == 3
        df["first_floor"] = df["floor"] == 1
        df["last_floor"] = df["floor"] == df["floors_count"]
        df.drop(columns=["floor", "rooms_count"], inplace=True)

        df.to_csv(root / "new_data.csv")
    
    @task
    def download_model():
        pass
    @task
    def predict_new_data():
        pass

    (
        get_data(n_rooms=1) >> get_data(n_rooms=2) >> get_data(n_rooms=3) >>
        preprocess_data() >>
        download_model() >>
        predict_new_data()
    )


dag = ProcessFlats()