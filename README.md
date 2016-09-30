# FacialExpressionRecognition

Built a convolutional neural network that achieves an accuracy of 65% on the Kaggle Dataset FER-2013,
matching the human accuracy of 65 ± 3%.
The dataset has around 35890 images of human faces labeled with one of the 7 expressions(0=Angry(13.5%) , 1=Disgust(1.5%), 2=Fear(14.3%), 3=Happy(25%), 4=Sad(16.8%), 5=Surprise(11%), 6=Neutral(17.3%)).

Looked at research papers, and adapted their models to this dataset. 
1 ModelCifarFinal.py is a model inspired by [Andrej Karpathy's model](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py) for Cifar-10.
It is a much deeper model with few tweaks and achieves an accuracy of 65%

2. ModelVGG contains code to implement transfer learning. The weight learned by VGGnet are applied to our dataset. 
It currently gets stuck at a local minima and predicts the most dominant class out of the 7 classes. 
Any suggestions, to improve it are most welcome.
