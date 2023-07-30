# covidat-tools

> Fix-fertige Auswertungen stehen zur Verfügung:
>
> * 📊 [monitoring](https://zeitferne.github.io/covidat-tools/export/monitoring.html):
>   SARS-CoV-2-Abwassermonitoring, Krankenstände, Sterbefälle, Medikamentenmangel.
>   Wird i.d.R. [täglich aktualisiert](.github/workflows/update-data.yaml) (ca. 6 Uhr morgens).
> * [covidat-old](https://zeitferne.github.io/covidat-tools/export/covidat-old.html):
>   Auswertungen zu COVID-19 aus den eingestellten EMS-Daten. Keine weiteren Aktualisierungen zu erwarten.
> * [Sterbetafeln](https://zeitferne.github.io/covidat-tools/export/Sterbetafeln.html):
>   Darstellungen von Lebenserwartungen & Veränderungen im Zeitverlauf.
>   Nur jährliche Aktualisierung der zugrundeliegenden Daten.

Tooling (Python scripts) and Jupyter/IPython notebooks for use with
[covidat-data](https://github.com/zeitferne/covidat-data).

This project uses [hatch](https://hatch.pypa.io/) as packaging & build system.

Refer to the daily update
[GitHub action workflow definition](.github/workflows/update-data.yaml)
for how to update & prepare the data. Use `hatch run notebooks:serve` to work
interactively with the notebooks & edit them, (instead of
the `python -m covidat.export` that the workflow uses).

## Interesting scripts

These might be the most interesting ones as they work directly with the data and
can also be useful if you plan to use R, Excel, etc. to analyze the data:

* [**dldata:**](src/covidat/dldata.py) This script is run on a daily schedule to collect files for
  covidat-data. Think twice if you really need to have your own duplicate of
  this job (but yes, theoretically I might miss some files, even though I do
  have a somewhat redundant setup with a cron job on a virtual server as well as
  local executions). You'll need to give the script an email address that it
  will send to all URLs it requests so that you can be contacted in case of
  troubles. The config used for covidat-data is in [`dldata.toml`](dldata.toml).

  This script is built to run without any library dependencies, for speedier
  execution of the "no-op" path in GitHub actions.
* [**collecthydro:**](src/covidat/collecthydro.py) Assembles collected abwassermonitoring data files into one
  csv file per data type (blverlauf, natmon_01)
* [**collectshortage:**](src/covidat/collectshortage.py) Summarizes collected basg-medicineshortage data into a
  single csv (count of limited products per day, usage (human/animal), and type
  of limitation)

These scripts read these environment variables (via [`util.py`](src/covidat/util.py)):

* `COVAT_DATA_ROOT`: Root directory for persistent data (dldata target,
  other's source). Directory should already exist (may be empty).
* `COVAT_COLLECT_ROOT`: Root directory for writing generated data to.
  Directory should already exist (may be empty).


Of (mostly historical) interest is also [`cmpdead.py`](src/covidat/cmpdead.py) which was
used to generate & post the tweets of <https://twitter.com/covidatTicker>.
