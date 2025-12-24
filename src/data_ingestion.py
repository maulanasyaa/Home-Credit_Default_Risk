import os

import pandas as pd
from sqlalchemy import create_engine

# conf db connection
db_connection_str = "postgresql://user:password@localhost:5432/home_credit_db"
db_engine = create_engine(db_connection_str)


# ingestion function
def upload_csv_to_postgres(folder_path):
    # read all files in folder
    files = [f for f in os.listdir(folder_path) if f.endswith("csv")]

    if not files:
        print("Files not found!")
        return

    print(f"Found {len(files)} data.")

    # create sql
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        table_name = filename.replace(".csv", "").lower()

        print(f"Processing file: {filename} -> table: {table_name}")

        try:
            df = pd.read_csv(file_path, low_memory=False)

            df.to_sql(
                table_name, db_engine, if_exists="replace", index=False, chunksize=10000
            )

            print(f"Success {filename} -> {len(df)} rows.")
        except Exception as e:
            print(f"Failed: {filename}. Error: {e}")

    print("Completed!")


if __name__ == "__main__":
    upload_csv_to_postgres("../data")
