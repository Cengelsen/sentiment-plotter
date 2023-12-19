## sentiment-plotter
This is a script that extracts discussion from the TC39's meeting notes and plots the sentiment of the discussion in a graph. 

### Dependencies

`Python 3.11`

### How to run locally

Currently (19.12.2023), you need to be running python version 3.11 to install the packages and run the script properly. I chose to use a conda environment, but any other virtual environment that facilitates for installation of python packages should work. It is recommended that you install the following packages in some virtual environment.

1. `pip install textblob tensorflow tensorrt git+https://github.com/LIAAD/yake tensorflow_hub pyqt5 pyqtwebengine jsonschema`

2. Place the script in a folder that contains the folders with meeting notes. The root should look like this:

    - root_folder
        - 2023-03/
            - mar-21.md
            - mar-22.md
            - mar-23.md
        - sentiment_plotter.py

3. `cd root_folder`
4. `python sentiment_plotter.py`

The runtime of the script is completely dependent on the amount of markdown files, and how many markdown headers are in the files.