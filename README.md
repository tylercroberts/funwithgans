# `funwithgans` README

### Installation Help:

To run these models, you will need to first install all dependencies.
These can be located in the requirements.txt files of the model folder you wish to run.

Move into the model folder and run the following command:

`pip install -r requirements.txt`

To utilize `make` alongside this, you will need to install `funwithgans` itself as a package.
You can do so from the root folder, using the following command which will install an editable version
of the package that you can make changes to and immediately see the effects.

`pip install -e .`


Make will allow you to run the examples with a simpler command, while limiting the options you can change.
They will oftentimes also perform additional steps, like cleaning up the output directories, 
or running other necessary stages.

In general, a `make` command will look something like:

`make dcgan-example`

You can pass additional arguments with `make` as follows:

`make dcgan-example STORAGE_DIR=../data MODEL_DIR='../models`

The subsections below will provide the required `make` commands, if there are any to run the examples.

To remove this package when you are done and keep your environment clean, use:

`pip uninstall funwithgans`


### Deep Convolutional Generative Adversarial Network (DCGAN):

(DCGAN Arxiv link here)[https://arxiv.org/abs/1511.06434]

To run this example, you will need to call `__init__.py` from the `dcgan` folder.
This can either be done directly, or through `make`. 

##### `make` command:

`make dcgan-example`

Additional Arguments include:
- STORAGE_DIR
- MODEL_DIR
- REPRODUCIBLE (if included, will always set the same seed)
- EPOCHS
- LR
- BETA (refers to `beta1` parameter of Adam optimizer)

If you require more customization than provided by the make command, you'll want to use python directly 
from your shell in order to run the example:

`python dcgan/src/__init__.py`

You can see a full list of possible command line arguments in the `parse_args` function contained in that file.



