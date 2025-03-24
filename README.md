# Inflow from a waster water treatment plant

This repository provides access to measurements of the water inflow to a waster water treatment plant (WWTP) from Denmark. The data is accompanied by measurements of relevant climate variables captured by a weather station near the WWTP.

The dataset contains approximately 18 months of hourly observations. The exact locations of the WWTP and weather station are confidential.

## Folder structure

The data in this repository are organized as follows

- `observations` contains all the relevant data. Each file in this folder contains measurements from a diferent source
- `processed` contains the matched data with minimal preprocessing as described [here](./code/0_overview.ipynb)
- `code` contains _jupyter_ notebooks and scripts to present and explore the data as well as helper functions used in those notebooks
    - You can refer to [here](.code/README.md) for more details on this section of the repository

Separately from the current repository, in [Zenodo]() you can find:
- [...]


## Data description

The dataset contains the following variables; all variables have been measured at hourly resolution.

| label | Description | Units | Provider |
| --- | --- | --- | --- |
| flow | Inflow of WWTP | m^3/h | - |
| acc_precip | Accumulated precipitation | mm | DMI |
| mean_pressure | Mean pressure | hPa | DMI |
| mean_radiation | Mean radiation (spectral range: 305-2800nm) | W/m^2 | DMI |
| mean_relative_hum | Mean relative humidity | % | DMI |
| mean_temp | Mean air temperature | °C | DMI |
| temp_grass | Temperature at grass height | °C | DMI |
| temp_soil_10 | Temperature at 10 cm underground | °C | DMI |
| temp_soil_30 | Temperature at 30 cm underground |  °C| DMI |

## Intended use

This repository is aimed at supporting educational, research, and exploratory activities, such as:

- __Experimenting__ with time series models for forecasting
- __Benchmarking__ machine learning models
    - In this [notebook](./code/1_data_split.ipynb), we suggest a data split to guarantee a consistent benchmark for new ML models
    - [Here](./code/2_baseline.ipynb) a simple baseline is established and [here](./code/example_modelling.ipynb) we include a modelling exercise


## Usage rights

DMI's free data is subject to the Creative Commons license CC BY. You can read more about the license terms [here](https://www.dmi.dk/friedata/guides-til-frie-data/vilkar-for-brug-af-data); in short, you are free to:

- Parts — copy and redistribute the material in any media or format for all purposes, including commercial.
- Adapt — remix, modify and build on the material for all purposes, including commercial ones.
- The Licensor may not revoke the freedoms as long as you follow the license terms.

Under the following conditions:

- Crediting - You must provide appropriate credit , provide a link to the license, and provide information on whether any changes have been made . You must do so in any sensible way, but not in a way that suggests that the licensor approves you or your use.
- No additional restrictions - You may not add legal terms or technological measures that legally restrict others from doing what the license allows.