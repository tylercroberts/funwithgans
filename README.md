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

The subsections below will provide the required `make` commands, if there are any to run the examples.

To remove this package when you are done and keep your environment clean, use:

`pip uninstall funwithgans`

### Deep Convolutional Generative Adversarial Network (DCGAN):

[DCGAN Arxiv link here](https://arxiv.org/abs/1511.06434)

To run this example, you will need to call `__init__.py` from the `dcgan` folder.
This can either be done directly with `python dcgan/src/__init__.py`, or through `make`. 

##### `make` command:

`make dcgan-example`

Using the make command will use a `config.json` file found in the `dcgan` directory in order to set flags 
and other command line arguments such as the location of necessary files. Please take a look at `parse_arguments` 
found in `dcgan/utils.py` to identify these arguments and create your config file accordingly.
Note that the names **MUST** be identical to those in the parser. (replace `-` with `_` in your `json`)




