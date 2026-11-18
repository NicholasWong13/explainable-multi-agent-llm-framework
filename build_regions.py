"""Build data/regions.csv from Google's public region-carbon-info dataset (2025 file).

Carbon columns are REAL (Google Cloud region carbon data, Apache-2.0 repo).
lat/lon/country are approximate and entered by hand (see GEO below); check them before publishing.
Regions whose location is unclear in the source (europe-west5, us-east7) are excluded on purpose.
Price is NOT included yet: no verified machine-readable price source has been collected.
"""
import csv
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("region-carbon-info/data/yearly/2025.csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/regions.csv")

# region -> (lat, lon, ISO country)  approximate site/city coordinates
GEO = {
    "africa-south1": (-26.20, 28.05, "ZA"), "asia-east1": (24.05, 120.52, "TW"),
    "asia-east2": (22.32, 114.17, "HK"), "asia-northeast1": (35.68, 139.69, "JP"),
    "asia-northeast2": (34.69, 135.50, "JP"), "asia-northeast3": (37.57, 126.98, "KR"),
    "asia-south1": (19.08, 72.88, "IN"), "asia-south2": (28.61, 77.21, "IN"),
    "asia-southeast1": (1.35, 103.82, "SG"), "asia-southeast2": (-6.21, 106.85, "ID"),
    "asia-southeast3": (13.76, 100.50, "TH"),
    "australia-southeast1": (-33.87, 151.21, "AU"), "australia-southeast2": (-37.81, 144.96, "AU"),
    "europe-central2": (52.23, 21.01, "PL"), "europe-north1": (60.57, 27.20, "FI"),
    "europe-north2": (59.33, 18.07, "SE"), "europe-southwest1": (40.42, -3.70, "ES"),
    "europe-west1": (50.47, 3.82, "BE"), "europe-west2": (51.51, -0.13, "GB"),
    "europe-west3": (50.11, 8.68, "DE"), "europe-west4": (53.44, 6.83, "NL"),
    "europe-west6": (47.38, 8.54, "CH"), "europe-west8": (45.46, 9.19, "IT"),
    "europe-west9": (48.86, 2.35, "FR"), "europe-west10": (52.52, 13.40, "DE"),
    "europe-west12": (45.07, 7.69, "IT"),
    "me-central1": (25.29, 51.53, "QA"), "me-central2": (26.43, 50.10, "SA"),
    "me-west1": (32.09, 34.78, "IL"),
    "northamerica-northeast1": (45.50, -73.57, "CA"), "northamerica-northeast2": (43.65, -79.38, "CA"),
    "northamerica-south1": (20.59, -100.39, "MX"),
    "southamerica-east1": (-23.55, -46.63, "BR"), "southamerica-west1": (-33.45, -70.67, "CL"),
    "us-central1": (41.26, -95.86, "US"), "us-central2": (36.15, -95.99, "US"),
    "us-east1": (33.20, -80.01, "US"), "us-east2": (33.75, -84.39, "US"),
    "us-east4": (39.04, -77.49, "US"), "us-east5": (39.96, -82.99, "US"),
    "us-south1": (32.78, -96.80, "US"), "us-west1": (45.60, -121.18, "US"),
    "us-west2": (34.05, -118.24, "US"), "us-west3": (40.76, -111.89, "US"),
    "us-west4": (36.17, -115.14, "US"),
}

rows, skipped = [], []
with open(SRC, newline="", encoding="utf-8") as f:
    for r in csv.reader(f):
        if r[0].startswith("Google Cloud Region"):
            continue
        region, loc, cfe, grid = r[0], r[1], r[2], r[3]
        if region not in GEO or not grid:
            skipped.append(region)
            continue
        lat, lon, cc = GEO[region]
        rows.append([region, loc, cc, lat, lon, cfe, grid])

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["region", "location", "country", "lat", "lon", "cfe", "grid_gco2_kwh"])
    w.writerows(rows)
print(f"wrote {len(rows)} regions to {OUT}; skipped: {skipped}")
