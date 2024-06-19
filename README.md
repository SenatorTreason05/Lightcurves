CIAO 4.15 must be installed as well as the other project dependencies.

In the correct conda env:
`python main.py`
or
`python main.py --no-gui`

The server for manipulation can be started with:
`python server.py`\
The user will be then prompted for an absolute path to the index file.\
Example: `/Users/mihirpatankar/Documents/Projects/Lightcurves/output/M3-2023-12-08_18:07:31/index.html`

You can also run this tool on a batch of objects.
First you must put your list of objects in `batch_run/objects_list.txt` (one object per line).
Then run `python batch_run/batch_run.py`
The batch progress will be continuously output to `batch_run/current_progress.txt`