# A Deep Q-Learning approach to Dino Run

This is the full repository for the project titled *A Deep Q-Learning approach to Dino Run* created for the Decision Models course @ Università degli Studi di Milano Bicocca A.Y 2019-20.

Refer to the `report.pdf` file for an in-depth methodological explanation.

## How to Run
Due to its high complexity, it is impossible to train the model on regular hardware. Even using a pre-trained model, the latency is too high on commodity hardware to be able to communicate efficiently with the web browser.

For this reason, a *Google Colaboratory* notebook was prepared, ready to run on a cloud Linux VM. To access it, click [here](https://colab.research.google.com/drive/17fsfDmrts2h-uQ4wrbZ7ntR-oGAZ1uX4). The notebook is well documented and nothing else is needed to run the code.

The `model/model.h5` file is a n HDF5 snapshot of the most recent model, and will be used in the notebook to ensure replicability.

## Dependencies

The Colab notebook is self-contained.
If you instead insist on running locally, be careful as the program dependends on a *Chromedriver* executable to interact with the web browser.
