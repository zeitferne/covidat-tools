# covidat-tools

> 📊 Die Notebooks inklusive Diagrammen werden unregelmäßig
> aktualisiert und sind hier abrufbar:
>
> * [monitoring](https://zeitferne.github.io/covidat-tools/export/monitoring.html):
>   SARS-CoV-2-Abwassermonitoring, Krankenstände, Medikamentenmangel. Wird (un)regelmäßig aktualisiert.
> * [covidat-old](https://zeitferne.github.io/covidat-tools/export/covidat-old.html):
>   Auswertungen zu COVID-19 aus den eingestellten EMS-Daten. Keine weiteren Aktualisierungen zu erwarten.
>
> Es ist auch der Code enthalten -- einfach Scrollen bis man zu den bunten
> Bildern kommt. Obwohl zugebenermaßen nicht alle ohne weiteren Kontext
> verständlich sind. Bei Fragen bitte Kontakt aufnehmen (über die Social Media
> Links im GitHub Profil oder auch einen Issue).
>
> Der tatsächliche Quellcode der Notebooks ist im
> [notebooks](notebooks/)-Verzeichnis.

Tooling (Python scripts) and Jupyter/IPython notebooks for use with
[covidat-data](https://github.com/zeitferne/covidat-data)

This project uses [hatch](https://hatch.pypa.io/) as packaging & build system.

Use `hatch run notebooks:serve` to serve the notebooks, set
the environment variables `COVAT_DATA_ROOT` and `COVAT_COLLECT_ROOT` before
(see below).

## Interesting scripts

These might be the most interesting ones as they work directly with the data and
can also be useful if you plan to use R, Excel, etc. to analyze the data:

* [**dldata:**](src/covidat/dldata.py) This script is run on a daily schedule to collect files for
  covidat-data. Think twice if you really need to have your own duplicate of
  this job (but yes, theoretically I might miss some files, even though I do
  have a somewhat redundant setup with a cron job on a virtual server as well as
  local executions). You'll need to give the script an email address that it
  will send to all URLs it requests so that you can be contacted in case of
  troubles. The config used for covidat-data is in [`dldata.toml`](dldata.toml)
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
