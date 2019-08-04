# Keras Scaffold TF2

The following is a scaffold for tensorflow 2.0 projects. It contains the following structure
 - trainer
   - utils
   - models
     - layers
     - unet_like
     - inpainting
   - datasets
   - callbacks
   config.py
   *.task.py
   
`models` contains keras models and layers. Currently only an inpainting model is provided
`datasets` contains functions which return a `tf.Dataset` object or other python iterable
`callbacks` contains keras callbacks used for training
`config.py` contains config hyperparameter variables
`*.task.py` are tasks that can be run from the command line to train a model

