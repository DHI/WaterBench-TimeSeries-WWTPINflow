# Inflow to a Waste Water Treatment Plant

 Accurate forecasting of the expected inflow—at least 12 hours in advance—is essential to optimize operations within a Wastewater Treatment Plant (WWTP). While traditionally employed for this purpose, mechanistic models ranging from simplified influent generators to complex sewer models face operational limitations due to either requiring continuous expert recalibration or suffering from inherent complexity and lengthy simulation times. The proliferation of online sensors monitoring WWTP influents, coupled with advancements in data-driven methods (e.g., machine learning), has created new opportunities for accurately characterizing and forecasting patterns in wastewater inflow dynamics. Data-driven approaches mainly rely on historical inflow data as target for training and recorded precipitation as key predictor in combined systems. Incorporating additional variables—such as time of year, time of day, air pressure, air temperature, and soil temperature— provides valuable context and potentially enhances the accuracy of the forecast.

This repository provides data to support the investigation of this problem, including measurements of the combined sewage inflow to a wastewater treatment plant (WWTP) in Denmark. The inflow data is accompanied by measurements of relevant meteorological variables captured by a weather station near the WWTP and provided by the Danish meteorological Institute (DMI). The dataset contains approximately 15 months of hourly observations. The exact location of the WWTP and weather station are omitted to anonymize the dataset.

In addition to the data, we provide benchmark metrics for the inflow prediction problem and illustrate with an example the intended use of the data.

## Folder structure

The data in this repository are organized as follows:

- `observations` contains all the measurements, split in two files respectively for the flow and meteorological data
- `processed` contains the matched data with minimal preprocessing as described [here](./code/0_overview.ipynb)
- `code` contains _jupyter_ notebooks and scripts to present and explore the data as well as helper functions used in those notebooks
    - More details on the code are provided in a separate [file](.code/README.md)

## Data description

The dataset contains the following variables, all measured at hourly resolution.

| label | Description | Units | Provider |
| --- | --- | --- | --- |
| flow | Inflow to WWTP | m<sup>3</sup>/h | (confidential) |
| acc_precip | Accumulated precipitation | mm | DMI |
| mean_pressure | Mean pressure | hPa | DMI |
| mean_radiation | Mean radiation (spectral range: 305-2800nm) | W/m<sup>2 | DMI |
| mean_relative_hum | Mean relative humidity | % | DMI |
| mean_temp | Mean air temperature | °C | DMI |
| temp_grass | Temperature at grass height | °C | DMI |
| temp_soil_10 | Temperature at 10 cm underground | °C | DMI |
| temp_soil_30 | Temperature at 30 cm underground |  °C| DMI |

Currently, this data set includes only measured variables. However, given the purpose of this repository, it will be relevant to include forecasted variables. <u>Our goal is to include weather forecast in future versions.</u>

## Intended use

This repository is aimed at supporting educational, research, and exploratory activities, such as:

- __Experimenting__ with time series models for forecasting
- __Benchmarking__ machine learning (ML) models

Importantly, the resources in the repository are limited to non-commercial purposes.

## Usage rights

This data is subject to the Creative Commons license CC BY. You can read more about the license terms [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en) or in DMI's free data [page](https://www.dmi.dk/friedata/guides-til-frie-data/vilkar-for-brug-af-data) (in danish). 

In short, you are free to:

- Parts - copy and redistribute the material in any media or format for all purposes, including commercial.
- Adapt - remix, modify and build on the material for all purposes, including commercial ones.
- The Licensor may not revoke the freedoms as long as you follow the license terms.

Under the following conditions:

- Crediting - You must provide appropriate credit , provide a link to the license, and provide information on whether any changes have been made. You must do so in any sensible way, but not in a way that suggests that the licensor approves you or your use.
- No additional restrictions - You may not add legal terms or technological measures that legally restrict others from doing what the license allows.

