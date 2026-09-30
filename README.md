# Input data

Each basin is one CSV file, `basin_<gauge>.csv`, with one row per dekad (October 1959 – December 1990, 1125 rows).

| Column | Meaning |
|---|---|
| `date` | first day of the dekad (1st, 11th or 21st) |
| `Q` | observed discharge, m³ s⁻¹ (CA-discharge) |
| `phase` | dekad of year |
| `T2M`, `PRCP`, `SNOW`, `MELT`, `SNOWF`, `PET`, `SWE_dif`, `DDsum` | basin-averaged ERA5-Land variables |
| `T2M_roll3`, `PRCP_roll3`, `MELT_roll3` | three-dekad rolling mean or sum |

Sources:
- Discharge: CA-discharge (Marti et al. 2023, Sci Data, https://doi.org/10.1038/s41597-023-02474-8)
- Forcing: ERA5-Land (Muñoz-Sabater et al. 2021), Copernicus Climate Change Service Climate Data Store

Place the nine CSV files here (or in `MyDrive/leakfree/basins/` when running in Colab).
