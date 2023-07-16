# covidat-tools

Tooling for use with [covidat-data](https://github.com/zeitferne/covidat-data).

This project started as a quick hack and was never properly cleaned up. Yes,
it's a complete and utter mess!

It does not define a proper Python package distribution, instead you use this
archaic workflow:

```shell
$ pip install -U -r requirements.txt
$ python src/SCRIPTNAME.py MORE ARGUMENTS
```

To use the modules from elsewhere (e.g. a Jupyter / IPython notebook) use this
extremely ugly hack like neither PyPI nor relative imports were invented yet:

```python
import sys, os.path
sys.path.append(os.path.abspath("path-to-src-folder"))
```

I do plan to add a proper `pyproject.toml` though. Eventually...

(And there is more to clean up. Please do not look at `cov.py` especially...)

## Interesting scripts

These might be the most interesting ones as they work directly with the data and
can also be useful if you plan to use R, Excel, etc. to analyze the data:

* [**dldata:**](src/dldata.py) This script is run on a daily schedule to collect files for
  covidat-data. Think twice if you really need to have your own duplicate of
  this job (but yes, theoretically I might miss some files, even though I do
  have a somewhat redundant setup with a cron job on a virtual server as well as
  local executions). You'll need to give the script an email address that it
  will send to all URLs it requests so that you can be contacted in case of
  troubles. The config used for covidat-data is in [`dldata.toml`](dldata.toml)
* **collecthydro:** Assembles collected abwassermonitoring data files into one
  csv file per data type (blverlauf, natmon_01)
* **collectshortage:** Summarizes collected basg-medicineshortage data into a
  single csv (count of limited products per day, usage (human/animal), and type
  of limitation)

These scripts read these environment variables (via [`src/util.py`](src/util.py)):

* `COVAT_DATA_ROOT`: Root directory for persistent data (dldata target,
  other's source). Directory should already exist (may be empty).
* `COVAT_COLLECT_ROOT`: Root directory for writing generated data to.
  Directory should already exist (may be empty).


Of (mostly historical) interest is also [`cmpdead.py`](src/cmpdead.py) which was
used to generate & post the tweets of <https://twitter.com/covidatTicker>.
