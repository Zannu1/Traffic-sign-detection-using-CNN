# Traffic-sign-detection-using-CNN
The training dataset contains around 39,000 images while the test dataset contains around 12,000 images containing 43 different classes. We will be using Convolutional Neural Networks(CNN) to solve this problem using Keras framework and TensorFlow as backend. Now Let’s start coding.

# Exploring dataset
In Windows, glob might give us mixed separators which will create a problem when we extract the class information from image paths, therefore, creating the below functions to fix the image paths.

#Preprocessing images
Now, we will use preprocess our images. We need preprocessing for two reasons:

To normalize intensity across all images i.e if an image is overexposed or underexposed, we will make it well-exposed using histogram equalization. As you can see in the above pictures, we have many such images.
To resize all images to the same size.

We will now preprocess images using preprocess_image function created above and store them in Numpy array and convert target images to One-Hot encoding for Keras usage.

#Building Convolutional Neural Network Model
We will now build our sequential CNN model with following specifications:

6 convolutional layer followed by one hidden layer and one output layer(fully connected or dense layer).
Dropout layers for regularization to avoid overfitting.
Relu activation function for all convolutional layers.
Softmax activation function for output layer as it is a Multi-class Classification problem.
Flatten layer for reshaping the output of the convolutional layer.

The learning rate scheduler will decay the learning rate. Slowing learning rates over epochs might help learn the model better.
The model checkpoint will save the model with the best validation accuracy as we progress. This will help in the case when our model starts overfitting.
Early Stopping will stop the training if the accuracy gain between ‘patience’ epochs is not more than a specified value.

#Training the built model
Training the model might take a long time using CPU. Because of the parallel nature of the neural networks, they work extremely well with GPUs(NVIDIA GPUs with CUDA). I used NVIDIA Tesla K80 which significantly decreased the training time by many folds.

#Improving Accuracy using Data Augmentation
This test accuracy is good but we can still improve it using Data augmentation, it increases the size of the training dataset by augmenting images using rotation, shearing, flipping, etc. effects.

#Conclusion
We created a Convolutional Neural Network (CNN) model to classify traffic sign images. We started by exploring our dataset of German traffic signs. Then we performed Pre-Processing of images (Histogram equalization and rescaling to the same size) to make them suitable for CNN. We built a simple CNN model using Keras with 6 convolutional layers followed by one hidden layer, one output layer(fully connected or dense layer). We used dropout layers to avoid overfitting. After that, we trained our model with our training dataset. The evaluation of the model resulted in 97.3% accuracy. We used data augmentation techniques to further improve accuracy to 98.4%. The human accuracy for this dataset is 98.84%. Pretty Close!
