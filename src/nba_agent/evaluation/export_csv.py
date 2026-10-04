"""Write every saved parquet file as a CSV in data/exports/ so it can be opened in Excel or Numbers."""
import pandas as pd

from nba_agent.data.tables import DATA, FILES

OUT = DATA / "exports"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        frame = pd.read_parquet(path)
        for col in frame.select_dtypes("datetime").columns:
            frame[col] = frame[col].dt.date
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(frame):>6} rows  {len(frame.columns):>2} columns")


if __name__ == "__main__":
    main()
