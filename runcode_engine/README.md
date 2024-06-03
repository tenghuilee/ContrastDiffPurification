# Run Code Engine

A simple utils to help running the scripts.

We write all tasks to a sqlite3 database, and run the task form the database.
Hope that it can help if you want to run the code to different machines.

The objective for this library is to provide a simplified interface to run code and collect results.

We will support:
- automatically analysis the run scripts and generate the run tasks
- automatically detect the number of GPU
- distribution of jobs across GPUs
- record the task status: 
  - running
  - completed
  - waiting

# How to use

```
cd runcode_engine
```

Update the code in `build_database.py` to add your own run scripts.
  
Run the file.
```
python build_database.py
```
