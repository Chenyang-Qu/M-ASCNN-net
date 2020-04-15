# M-ASCNN-net
This project is the code of the thesis "Multi-spectral remote sensing image classification method based on convolutional neural networks"

This project is a convolutional neural network project for semantic segmentation of multi-spectral remote sensing images，You can use your own data set to run this project code to complete the semantic segmentation task.
# getallbands.py
The getallbands.py file is a subroutine that generates a data set,You can use your own multi-spectral remote sensing images and label images to generate data sets.

Make the remote sensing image and the label image have the same name，put it in the root folder of the project.

Enter the following command on the command line：
python getallbands.py -ch xx

XX is the number of channels of the multispectral image, you need to tell the code how many spectral bands your original image has.

After the execution of this file is completed, a segmented remote sensing image set and label set will be generated.

# train.py
The train. file is a training file for convolutional neural networks. This file uses the generated data set for training. The specific structure of the network is shown in netmodel.png.

Run the network model training program train.py. The model is saved as "NETNAME". The default values are used for epoch, batch size, and learning rate. Command line input:
python train.py -m NETNAME -ch XX

NETNAME and the number of channels can be replaced with values suitable for your data set.

Please adjust the training batch size according to the GPU memory size and the number of image channels.

The number of epochs is 30 by default.

# predict.py
The predict.py file can generate a graph of predicted results. Put the remote sensing image to be classified into the catalog and run this program.

python predict.py -ch XX
XX is the number of channels of the multispectral image.
