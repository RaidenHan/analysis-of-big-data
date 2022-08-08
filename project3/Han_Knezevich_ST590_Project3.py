# %%
###############################################################################
# Introduction ################################################################
###############################################################################
"""
In this project, we'll use the Detecting Heavy Drinking Data Set from the UCI
machine learning repository to simulate stream data generation and perform
some basic PySpark ETL operations. Specifically, we will create a PySpark
Session in Python and perform data processing. For readers running the
program directly in PySpark, please skip the PySpark Startup section.
"""

# Module preparation
import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
from tqdm import trange
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col

# %%
###############################################################################
# PySpark Startup #############################################################
###############################################################################
"""
To begin with, since this program will be running on Windows and is affected
by the virtual environment configuration, we need to set up temporary
environment variables to ensure the smooth operation of PySpark. Also, the new
Spark session is set to use four CPU cores for parallel computation.
"""

# Prepare the environment for Windows operating systems
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Create a new Spark session
spark = SparkSession.builder.master(
    "local[4]").appName("my_app").getOrCreate()

# %%
###############################################################################
# Set Up for Creating Files ###################################################
###############################################################################
"""
In this section, we will design a function. This function will read the
original dataset and gradually generate the decomposed data according to the
specified chunk size, delayed time, and the number of iterations.
"""


def create_stream_files(pids, chunksize=500, delay=20, iteration=15):
    """ Simulate stream data in separate folders using the Detecting Heavy
    Drinking Data Set from UCI machine learning repository

    Parameters
    ----------
    pids : list
        A list containing all PID strings of interest
    chunksize : int
        The number of values per iteration
    delay : int or float
        Waiting time in seconds before proceeding to the next loop
    iteration : int
        The number of iterations

    """

    # Read the original dataset
    filename = r'Data\all_accelerometer_data_pids_13.csv'
    heavy_drinking = pd.read_csv(filename)
    # Convert the unix timestamp to datetime
    heavy_drinking['time'] = pd.to_datetime(heavy_drinking['time'], unit='ms')
    # Select data for specific users
    dfs = [heavy_drinking[heavy_drinking['pid'] == pid] for pid in pids]
    # Use an array to denote whether the data is exhausted or not
    exhausted = np.full(len(dfs), False)
    # Generate batch data
    for i in trange(iteration):
        for j in range(len(pids)):
            if not exhausted[j]:
                try:
                    # Save one batch to a .csv file
                    dfs[j].iloc[i * chunksize:(i + 1) * chunksize, :].to_csv(
                        fr'Data\{pids[j]}\{pids[j]}_{i}.csv', index=False)
                except IndexError:
                    # Output to the console if the data is exhausted
                    print(f"{pids[j]} data is exhausted!")
                    exhausted[j] = True
        # Delay
        time.sleep(delay)

    return


def create_empty_directory(pids):
    """ Create and empty directories to ensure the program is repeatable

    Parameters
    ----------
    pids : list
        A list containing all PID strings of interest

    """

    # Set up the stream data storage directory
    for pid in pids:
        os.makedirs(fr'Data\{pid}', exist_ok=True)
        # Delete all existing files
        for filename in os.listdir(fr'Data\{pid}'):
            os.remove(os.path.join(fr'Data\{pid}', filename))
    # Empty the checkpoint and the destination directory
    shutil.rmtree('Checkpoint', ignore_errors=True)
    shutil.rmtree('Destination', ignore_errors=True)

    return


# %%
###############################################################################
# Reading a Stream ############################################################
###############################################################################
"""
In this section, we will configure the target folder for the stream data to be
read and start that read process.
"""

# Set up the stream data directory and delete previous results
pids = ['SA0297', 'PC6771']
create_empty_directory(pids)
# Specify the format of each column of the CSV file
struct = StructType([StructField('time', TimestampType(), False),
                     StructField('pid', StringType(), False),
                     StructField('x', FloatType(), False),
                     StructField('y', FloatType(), False),
                     StructField('z', FloatType(), False)])
# Read the stream
hd_sa0297 = spark.readStream.schema(struct).option(
    'header', 'true').csv(r'Data\SA0297')
hd_pc6771 = spark.readStream.schema(struct).option(
    'header', 'true').csv(r'Data\PC6771')

# %%
###############################################################################
# Transform/Aggregation Step ##################################################
###############################################################################
"""
In this section, we will calculate the L2 norm of the x,y,z columns and 
integrate the time and pid information into a separate table.
"""

hd_sa0297_l2 = hd_sa0297.select('time', 'pid', (
        (col('x') ** 2 + col('y') ** 2 + col('z') ** 2) ** 0.5).alias('L2'))
hd_pc6771_l2 = hd_pc6771.select('time', 'pid', (
        (col('x') ** 2 + col('y') ** 2 + col('z') ** 2) ** 0.5).alias('L2'))

# %%
###############################################################################
# Writing the Streams #########################################################
###############################################################################
""" 
This section will write each stream to its .csv file, including a checkpoint 
location. 
"""

# Set the output of the data stream and start the queries
write_sa0297 = hd_sa0297_l2.writeStream.outputMode('append').format(
    'csv').option('checkpointlocation', r'Checkpoint\SA0297').option(
    'path', r'Destination\SA0297').option('header', 'true').start()
write_pc6771 = hd_pc6771_l2.writeStream.outputMode('append').format(
    'csv').option('checkpointlocation', r'Checkpoint\PC6771').option(
    'path', r'Destination\PC6771').option('header', 'true').start()
# Start to generate data streams
create_stream_files(pids)

# Stop the queries
time.sleep(10)  # Allow some time to process the last batch
write_sa0297.stop()
write_pc6771.stop()

# %%
###############################################################################
# Parsing the .csv Files ######################################################
###############################################################################
""" 
Finally, we will need to find a way to deal with the data in the .csv files we 
have written. We will read all pieces and output data for each person to a 
single .csv file. 
"""

# Read in all part files
allfiles_sa0297 = spark.read.option('header', 'true').csv(
    r'Destination\SA0297\part-*.csv')
allfiles_pc6771 = spark.read.option('header', 'true').csv(
    r'Destination\PC6771\part-*.csv')
# Output as a single .csv file
allfiles_sa0297.coalesce(1).write.format('csv').option(
    'header', 'true').save(r'Destination\SA0297\single_csv_file')
allfiles_pc6771.coalesce(1).write.format('csv').option(
    'header', 'true').save(r'Destination\PC6771\single_csv_file')

# Close the PySpark session
spark.stop()
