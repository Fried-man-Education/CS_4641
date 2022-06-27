# Environment Setup

1. Refer to the EdStem post about [VS Code and Anaconda Setup Guide](https://edstem.org/us/courses/22268/discussion/1575722) if you do not already have Anaconda or Miniconda installed.
2. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Use the command: `conda env create -f ml_hw2_env_<OS>.yml`
    - Make sure to modify the command based on your OS (`linux`, `mac`, or `win`).
    - This should create an environment named `ml_hw2`. 
3. Activate the conda environment:
    - Windows command: `activate ml_hw2` 
    - MacOS / Linux command: `conda activate ml_hw2`

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)