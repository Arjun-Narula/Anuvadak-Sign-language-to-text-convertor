# ANUVADAK-Sign-language-to-text-convertor
## CONTENTS 
<br>[Abstract ](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#abstract)
<br>[Chapter 1 Introduction](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#chapter-1-introduction)
<br>[Chapter 2 Literature Survey](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#chapter-2-literature-survey)
<br>[Chapter 3 Artificial Neural Network – A Review](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#chapter-3-artificial-neural-network--a-review)
<br>[Chapter 4 Methodology ](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#chapter-4-methodology)
<br>[Chapter 5 Results and Applications ]()
<br>[Chapter 6 Conclusions and Future Scope ]
<br>[References ]
<br>[Appendix]

## ABSTRACT
People affected by speech impairment can't communicate using hearing
and speech, they rely on sign language for communication. Sign
language is used among everybody who is speech impaired, but they
find a hard time communicating with people which are nonsigners
(people aren’t proficient in sign language). So requirement of a sign
language interpreter is a must for speech impaired people. This makes
their informal and formal communication difficult. There has been
favorable progress in the field of gesture recognition and motion
recognition with current advancements in deep learning. The proposed
system tries to do a real time translation of hand gestures into equivalent
English text. This system takes hand gestures as input through video
and translates it text which could be understood by a non-signer. There
were similar researches done earlier with most of them focussing on just
sign translation of English alphabets or just numbers. There will be use
of CNN for classification of hand gestures. By deploying this system, the
communication gap between signers and non-signers. This will make
communication speech impaired people less cumbersome. There were
many researches which helped us to establish the idea of Artificial
Neural Networks for this project. Many models were available that
detected only characters with an accuracy of around 86%. We also
explored the Linear discriminant analysis(LDA) technique but didn’t use
it due its drawback to express complex data. The hardware
implementation of the project was also read about. It has a cost and
maintenence factor involved which we tried to eliminate in our project.
We have achieved high acccuracy of 96.5% in our model, with the
feature of suggestions of words and formation of sentences, an idea
which was not found in any of the researches.

## CHAPTER-1 INTRODUCTION
American sign language is a predominant sign language Since the
only disability D&M people have is communication related and they
cannot use spoken languages hence the only way for them to
communicate is through sign language.<br />
Communication is the process of exchange of thoughts and
messages in various ways such as speech, signals, behaviour and
visuals. Deaf and dumb(D&M) people make use of their hands to
express different gestures to express their ideas with other people.<br />
Gestures are the nonverbally exchanged messages and these
gestures are understood with vision. This nonverbal communication
of deaf and dumb people is called sign language.<br />
Sign language is a visual language and consists of 3 major
components:<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Table1.1%20Components%20of%20visual%20sign%20language.JPG">
</p><br />


In our project we basically focus on producing a model which can
recognise Finger spelling based hand gestures in order to form a
complete word by combining each gesture.<br />
The gestures we aim to train are as given in the image below.<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%201.1%20Sign%20conventions%20of%20alphabets%20in%20ASL.JPG">
 </p><br /> 
  



### 1.1 MOTIVATION
The main motivation is to translate sign language to text. The
framework provides a helping-hand for speech-impaired to
communicate with the rest of the world using sign language. This
leads to the elimination of the middle person who generally acts as a
medium of translation. This would contain a user-friendly environment
for the user by providing text output for a sign gesture input.<br />
There are many researches on recognition of sign languages or
finger spellings. In [1] they have proposed a system that recognizes
dynamic hand gestures for English numbers (0-9) in real-time using
Hidden Markov Model. HMM is highly dependent on the probability of
the hidden states, hence there are more number of parameters to
learn which is time consuming. Our prime motivation was to devise a
system which is time efficient for each and every alphabet of
American Sign Language.<br />

Pradumn Kumar and Upasana Dugal [2] proposed an algorithm using
Tensorflow based on Advanced Convolutional Neural Networks for
identification of plants. This research paper motivated us to use
Convolutional Neural Networks for identification of American Sign
Language symbols.<br />
In [3], [4] and [5] we got an overview of Convolutional Neural
Network(CNN), various deep learning approaches, their applications
and their recent developments. We also gathered details about how
overfitting in Neural Networks can be reduced by a technique called
Dropout which is dropping off some neurons in the Neural Nets.<br />
One method is based on Support Vector Machine (SVM) [7]. They
extract signs from video sequences using skin colour segmentation,
and distinguish static and dynamic gestures. However, the accuracy
received in this method is 86%. Our aim is to achieve higher
accuracy than this model.<br />
Representing high-volume and high-order data is an essential
problem, especially in machine learning field. This [8] research paper
developed a model using Linear discriminant analysis (LDA), the
classical LDA, however, demands that input data should be
represented by vector. Such a constraint is a significant drawback to
express complex data. To solve the above limitation for high-volume
and high-order data nonlinear dimensionality reduction, we used
Convolutional Neural Networks.<br />
Another method is a controller gesture recognition system that
extracts hand gestures using Flex sensors [9] for sensing the hand
movements. The output from the microcontroller is the recognized
text which is fed as input to the speech synthesizer. Arduino
microcontroller processes the data for each particular gesture made.
The system is trained for different voltage values for each letter.
However, this method is not very economical and requires high
maintenance from time to time.<br />
We propose to develop a user friendly human computer interface
(HCI) where the computer understands the human sign language and
is accurate, effective and economical.<br />

### 1.2 OBJECTIVES
* To develop an application interface that interprets American Sign
  Language to Text in real time to assist deaf and dumb for
  communicating with them effectively eliminating the requirement of a
  translating individual.<br />
* To devise a model to achieve the highest possible accuracy and least
  time consumption for prediction of symbols as compared to already
  existing models.<br />
* To reduce the cost and develop an economical and user friendly
  graphical user interface (GUI) application that requires minimal
  maintenance for conversion of sign to its corresponding text.<br />
* To provide suggestions based on current word to eliminate the need
  of translating the full word, thereby improving accuracy and reducing
  time for sign to text conversion.<br />
* To reduce the chances of spelling mistakes by suggesting correct
  spellings from English dictionary words close to the current word.
* To formulate characters, words and sentences with interpretation of
  symbols in American Sign Language in a portable and real time
  application.<br />
  
  ## Chapter 2 Literature Survey 
  There are many researches on recognition of sign languages or
finger spellings. We introspected various reasearch papers that were
available and came up with an interface that converts sign language
to text, provides the feature of adding a word and suggestions based
on the word being translated.<br />
In [1], they have proposed a system that recognizes dynamic hand
gestures for English numbers (0-9) in real-time using Hidden Markov
Model. HMM is highly dependent on the probability of the hidden
states, hence there are more number of parameters to learn which is
time consuming. The system contains two stages: Preprocessing for
hand tracking and Classification to recognize gestures. Hidden
Markov Model is used for the isolated and dynamic gesture
recognition whose average recognition rates are 99.167% and
93.84% respectively. Hidden Markov Models (HMM) is used for the
classification of the gestures. This model deals with dynamic aspects
of gestures.Gestures are extracted from a sequence of video images
by tracking the skin-colour blobs corresponding to the hand into a
body– face space centered on the face of the user. The goal is to
recognize two classes of gestures: deictic and symbolic.The image is
filtered using a fast look–up indexing table. After filtering, skin colour
pixels are gathered into blobs.<br />
Pradumn Kumar and Upasana Dugal [2] proposed an algorithm using
Tensorflow based on Advanced Convolutional Neural Networks for
identification of plants. This research paper motivated us to use
Convolutional Neural Networks for identification of American Sign
Language symbols. Specially using CNN is a very trending procedure
for Deep learning in computer point of view. ImageNet have produced
a lot of expectation by giving exciting results. Here CNN takes the
most challenging task for identification of plants by using their
complete picture or any parts of that plants while others tackles one
by one process like firstly they take any specific organisms (flowers,
leaves and bark etc.) then whole picture of organisms. In CNN there
are some limitations like it
is not better with very large sets of images or lack of explanatory
power. So Advanced CNN will replace CNN because in Advanced
CNN is small in size as compare to CNN for recognizing images.
Here large models can be easily scale up and these models are small
enough to train fast, by this we will get out new ideas and have a
good chance for experiment on other methods also. The architecture
of Advanced CNN is multi-layer consisting of alternate use of
Convolution layers and nonlinearities. All these layers are followed by
fully connected layers leading into a softmax classifier. This model
gives a good accuracy results with in few time when we run on a
GPU.<br />
[3] presents a comprehensive review of deep learning and develops
a categorization scheme to analyze the existing deep learning
literature. It divides the deep learning algorithms into four categories
according to the basic model they derived from: Convolutional
Neural Networks, Restricted Boltzmann Machines, Autoencoder and
Sparse Coding. The state-of-the-art approaches of the four classes
are discussed and analyzed in detail. For the applications in the
computer vision domain, the paper mainly reports the
advancements of CNN based schemes, as it is the most extensively
utilized and most suitable for images. Most notably, some recent
articles have reported inspiring advances showing that some CNNbased
algorithms have already exceeded the accuracy of human
raters.Despite the promising results reported so far, there is
significant room for further advances. For example, the underlying
theoretical foundation does not yet explain under what conditions
they will perform well or outperform other approaches, and how to
determine the optimal structure for a certain task. This paper
describes these challenges and summarizes the new trends in
designing and training deep neural networks, along with several
directions that may be further explored in the future.<br />
Convolution neural network has long been used in the field of
digital image processing and speech recognition, and has
achieved great success. Before the convolutional neural network
was proposed, both image processing and speech recognition were
done by traditional machine learning algorithms. Although great
results were achieved, it was difficult to make further breakthroughs,
so CNN came into being. Currently, CNN for image processing and
speech recognition are
relatively mature. Both the theoretical research and the industrial
application have been very successful, which has promoted
CNN's leap-forward development. CNN's success of image
processing and speech recognition has stimulated its research
frenzy in natural language processing. The current CNN to handle
natural language has been widely used, although some
achievements have been made, the current effect is not very good.
The purpose of [4] is to give a clearer explanation of the structure of
CNN. At the same time, give a brief summary and prospect of
current CNN research in image processing, speech recognition and
natural language processing.<br />
Results observed in the comparative study with other traditional
methods suggest that CNN gives better accuracy and boosts the
performance of the system due to unique features like shared
weights and local connectivity.<br />
CNN is better than other deep learning methods in applications
pertaining to computer vision and natural language processing
because it mitigates most of the traditional problems.<br />
To reduce the problem of overfitting, we used the Dropout Technique
as suggested in [5]. Dropout is a technique for improving neural
networks by reducing overfitting. Standard backpropagation learning
builds up brittle co-adaptations that work for the training data but do
not generalize to unseen data. Random dropout breaks up these coadaptations
by making the presence of any particular hidden unit
unreliable. This technique was found to improve the performance of
neural nets in a wide variety of application domains including object
classification, digit recognition, speech recognition, document
classification and analysis of computational biology data. This
suggests that dropout is a general technique and is not specific to
any domain. Dropout considerably improved the performance of
standard neural nets on other data sets as well. This idea can be
extended to Restricted Boltzmann Machines and other graphical
models. The central idea of dropout is to take a large model that
overfits easily and repeatedly sample and train smaller sub-models
from it.<br />
Ankit Ojha, Ayush Pandey, Shubham Maurya, Abhishek Thakur and
Dr. Dayananda P in [6] developed a finger spelling sign language
translator is obtained which has an accuracy of 95%. They created a
desktop application that uses a computer’s webcam to capture a
person signing gestures for ASL, and translate it into corresponding
text and speech in real time. The translated sign language gesture
will be acquired in text which is farther converted into audio. In this
manner they are implementing a finger spelling sign language
translator. To enable the detection of gestures, they used
Convolutional neural network (CNN). This research paper provided
us insight about the base model for our project.<br />
In the following paper [7], Support Vector Machine (SVM) was used
as the machine learning method. They proposed a recognition
method of fingerspellings in Japanese sign language, which uses
classification tree based on pattern recognition and machine learning.
Fingerspellings of Japanese sign language are based on American
alphabets, and some are added according to Japanese Character,
gestures, numbers, and meanings. They constructed a classfication
tree for easily recognized fingerspellings and also used machine
learning for dificultly recognized ones. They achieved an accuracy of
86% using this model.<br />
Representing high-volume and high-order data is an essential
problem, especially in machine learning field. The [8] research paper
developed a model using Linear discriminant analysis (LDA). The
classical LDA, however, demands that input data should be
represented by vector. Such a constraint is a significant drawback to
express complex data. In this paper, they have proposed a
convolutional 2D-LDA method for nonlinear dimensionality reduction.
The difficult problem of optimization is solved by a clever equivalence
of two objective functions. The proposed method employs a two
stage end-to-end CNN to realize dimensionality reduction.
Effectiveness of such structure has been proved with two different
networks. The convolutional 2D LDA method out- performs the
classical LDA in all experiment settings.<br />
Another method is a hardware based controller gesture recognition
system that extracts hand gestures using Flex sensors [9] for sensing
the hand movements using a glove. The sensor glove design along
with the tactile sensor helps in reducing the ambiguity in gestures
and shows improved accuracy. The output from the microcontroller is
the recognized text which is fed as input to the speech synthesizer.
Arduino microcontroller processes the data for each particular
gesture made. The system is trained for different voltage values for
each letter. However, this method is not very economical and
requires high maintenance from time to time.<br />
## Chapter 3 Artificial Neural Network – A Review 
<b>*Feature Extraction and Representation </b> : The representation of an image
as a 3D matrix having dimension as of height and width of the image and the
value of each pixel as depth ( 1 in case of Grayscale and 3 in case of RGB).
Further, these pixel values are used for extracting useful features using
CNN.*<br />

# 3.1 Artificial Neural Networks
Artificial Neural Network is a connections of neurons, replicating
the structure of human brain. Each connection of neuron transfers
information to another neuron. Inputs are fed into first layer of
neurons which processes it and transfers to another layer of
neurons called as hidden layers. After processing of information
through multiple layers of hidden layers, information is passed to
final output layer.<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.1%20Artificial%20Neural%20Network.JPG">
</p><br />
There are capable of learning and they have to be trained. There are
different learning strategies :<br />

* Unsupervised Learning<br />
* Supervised Learning<br />
* Reinforcement Learning<br />

### 3.1.1. Unsupervised Learning:
Unsupervised learning is a type of machine learning that looks for
previously undetected patterns in a data set with no pre-existing
labels and with a minimum of human supervision. Two of the main
methods used in unsupervised learning are principal component and
cluster analysis.<br />
The only requirement to be called an unsupervised learning strategy
is to learn a new feature space that captures the characteristics of the
original space by maximizing some objective function or minimising
some loss function. Therefore, generating a covariance matrix is not
unsupervised learning, but taking the eigenvectors of the covariance
matrix is because the linear algebra eigendecomposition operation
maximizes the variance; this is known as principal component
analysis.<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.2%20Unsupervised%20learning.JPG">
</p><br />

### 3.1.2. Supervised Learning:
Supervised learning is the machine learning task of learning a
function that maps an input to an output based on example
input-output pairs. It infers a function from labeled training data
consisting of a set of training examples. In supervised learning,
each example is a pair consisting of an input object (typically a
vector) and a desired output value (also called the supervisory
signal). A supervised learning algorithm analyzes the training
data and produces an inferred function, which can be used for
mapping new examples. An optimal scenario will allow for the
algorithm to correctly determine the class labels for unseen
instances. This requires the learning algorithm to generalize
from the training data to unseen situations in a "reasonable"
way.<br/>
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.3%20Supervised%20learning.JPG">
</p><br />

### 3.1.3. Reinforcement Learning:
Reinforcement learning (RL) is an area of machine learning
concerned with how software agents ought to take actions in an
environment in order to maximize the notion of cumulative reward.
Reinforcement learning is one of three basic machine learning
paradigms, alongside supervised learning and unsupervised learning.
Reinforcement learning differs from supervised learning in not
needing labelled input/output pairs be presented, and in not needing
sub-optimal actions to be explicitly corrected. Instead the focus is on
finding a balance between exploration (of uncharted territory) and
exploitation (of current knowledge).<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.4%20Reinforcement%20learning.JPG">
</p><br />

### 3.2 Convolution Neural Network
Unlike regular Neural Networks, in the layers of CNN, the neurons are
arranged in 3 dimensions: width, height, depth. The neurons in a layer
will only be connected to a small region of the layer (window size)
before it, instead of all of the neurons in a fully-connected manner.
Moreover, the final output layer would have dimensions (number of
classes), because by the end of the CNN architecture we will reduce
the full image into a single vector of class scores.<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.5%20Convolution%20Neural%20Network.JPG">
</p><br />

#### 1 Convolution Layer :
In convolution layer we take a small window size (typically of
length 5*5) that extends to the depth of the input matrix. The
layer consist of learnable filters of window size. During every
iteration we slid the window by stride size (typically 1), and
compute the dot product of filter entries and input values at a
given position. As we continue this process well create a 2-
Dimensional activation matrix that gives the response of that
matrix at every spatial position. That is, the network will learn
filters that activate when they see some type of visual feature
such as an edge of some orientation or a blotch of some
color.<br />

#### 2 Pooling Layer :
We use pooling layer to decrease the size of activation
matrix and ultimately reduce the learnable parameters. There
are two type of pooling:<br />

##### 2.a Max Pooling :
In max pooling we take a window size (for
example window of size 2*2), and only take the maximum of 4
values. Well lid this window and continue this process, so well
finally get a activation matrix half of its original Size.<br />

##### 2.b Average Pooling :
In average pooling we take average of all
values in a window.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.6%20Average%20Pooling%20and%20Max%20Pooling.JPG">
</p><br />

#### 3 Fully Connected Layer :
In convolution layer neurons are connected only to a local region,
while in a fully connected region, well connect the all the inputs to
neurons.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.7%20Fully%20converted%20Layer.JPG">
</p><br />


#### 4 Final Output Layer :
After getting values from fully connected layer, well connect
them to final layer of neurons(having count equal to total
number of classes), that will predict the probability of each
image to be in different classes.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.8%20Output%20Layer.JPG">
</p><br />

### 3.3 TensorFlow
Tensorflow is an open source software library for numerical
computation. First we define the nodes of the computation graph,
then inside a session, the actual computation takes place.
TensorFlow is widely used in Machine Learning.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.9%20Tensorflow.JPG">
</p><br />


### 3.4 Keras
Keras is a high-level neural networks library written in python that
works as a wrapper to TensorFlow. It is used in cases where we want
to quickly build and test the neural network with minimal lines of code.
It contains implementations of commonly used neural network
elements like layers, objective, activation functions, optimizers, and
tools to make working with images and text data easier.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.10%20Keras.JPG">
</p><br />


### 3.5 OpenCV
OpenCV(Open Source Computer Vision) is an open source library of
programming functions used for real-time computer-vision. It is mainly
used for image processing, video capture and analysis for features
like face and object recognition. It is written in C++ which is its
primary interface, however bindings are available for Python, Java,
MATLAB/OCTAVE.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%203.11%20OpenCV.JPG">
</p><br />

## Chapter 4 Methodology 
*The system is a vision based approach. All the signs are represented
with bare hands and so it eliminates the problem of using any artificial
devices for interaction.*<br />

### 4.1 Data Set Generation
For the project we tried to find already made datasets but we couldn’t
find dataset in the form of raw images that matched our requirements.
All we could find were the datasets in the form of RGB values. Hence
we decided to create our own data set. Steps we followed to create
our data set are as follows:<br />

#### Step 1:
We used Open computer vision(OpenCV) library in order to
produce our dataset. Firstly we captured around 600 images
of each of the symbol in ASL for training purposes and
around 150 images per symbol for testing purpose.<br />
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%204.1%20Capturing%20Image%20for%20a%20dataset.JPG">
</p><br />



#### Step 2:
First we capture each frame shown by the webcam of our
machine. In the each frame we define a region of interest
(ROI) which is denoted by a blue bounded square as shown
in the image below.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%204.2%20RGB%20image%20of%20alphabet%20%E2%80%9CA%E2%80%9D.JPG">
</p><br />

#### Step 3:
From this whole image we extract our Region of Interest
(ROI) which is RGB and convert it into gray scale Image as
shown below.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%204.3%20Grayscale%20image%20of%20alphabet%20%E2%80%9CA%E2%80%9D.JPG">
</p><br />

#### Step 4:
Finally we apply our gaussian blur filter to our image which
helps us extracting various features of our image. The
image after applying gaussian blur looks like below.

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%204.4%20Image%20after%20Gaussian%20Filter%20.JPG">
</p><br />

### 4.2 GESTURE CLASSIFICATION
<b>The approach which we used for this project is :</b><br />
Our approach uses two layers of algorithm to predict the final symbol of the
user.<br />
#### 4.2.1 Algorithm Layer :

* Apply gaussian blur filter and threshold to the frame taken with
opencv to get the processed image after feature extraction.<br />
* This processed image is passed to the CNN model for prediction
and if a letter is detected for more than 60 frames then the letter is
printed and taken into consideration for forming the word.<br />
* Space between the words are considered using the blank symbol.<br />

#### 4.2.2 CNN Model :
* <b>1st Convolution Layer :</b>  
The input picture has resolution of 128x128
pixels. It is first processed in the first convolutional layer using 32 filter
weights (3x3 pixels each). This will result in a 126X126 pixel image, one
for each Filter-weights.<br />


* <b> 1st Pooling Layer : </b> 
The pictures are downsampled using max pooling of
2x2 i.e we keep the highest value in the 2x2 square of array. Therefore,
our picture is downsampled to 63x63 pixels.<br />

* <b> 2nd Convolution Layer : </b>
Now, these 63 x 63 from the output of the first
pooling layer is served as an input to the second convolutional layer.It is
processed in the second convolutional layer using 32 filter weights (3x3
pixels each).This will result in a 61 x 61 pixel image.<br />
* <b>2nd Pooling Layer : </b>
The resulting images are downsampled again using
max pool of 2x2 and is reduced to 30 x 30 resolution of images.<br />
* <b> 1st Densely Connected Layer :</b> 
Now these images are used as an input
to a fully connected layer with 128 neurons and the output from the
second convolutional layer is reshaped to an array of 30x30x32 =28800
values. The input to this layer is an array of 28800 values. The output of
these layer is fed to the 2nd Densely Connected Layer.We are using a
dropout layer of value 0.5 to avoid overfitting.<br />

* <b> 2nd Densely Connected Layer :<b>
  Now the output from the 1st Densely
Connected Layer are used as an input to a fully connected layer with 96
neurons.<br />
  * <b> Final layer:</b>
  The output of the 2nd Densely Connected Layer serves as
an input for the final layer which will have the number of neurons as the
number of classes we are classifying (alphabets + blank symbol).<br />
  
  
<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%204.5%20The%20CNN%20model%20used%20in%20the%20project.JPG">
</p><br />

  

### 4.2.3 Activation Function :
We have used ReLu (Rectified Linear Unit) in each of the
layers(convolutional as well as fully connected neurons). ReLu
calculates max(x,0) for each input pixel. This adds nonlinearity to the
formula and helps to learn more complicated features.It helps in
removing the vanishing gradient problem and speeding up the training
by reducing the computation time.<br />

### 4.2.4 Pooling Layer :
We apply Max pooling to the input image with a pool size of (2, 2)
with relu activation function.This reduces the amount of parameters
thus lessening the computation cost and reduces overfitting.<br />

### 4.2.5 Dropout Layers:
The problem of overfitting, where after training, the weights of the
network are so tuned to the training examples they are given that the
network doesn’t perform well when given new examples.This layer
“drops out” a random set of activations in that layer by setting them to
zero.The network should be able to provide the right classification or
output for a specific example even if some of the activations are
dropped out.<br />

<p align="center">
  <img width="460" height="300" src="https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/Images/Figure%204.6%20.JPG">
</p><br />

### 4.2.6 Optimizer :
We have used Adam optimizer for updating the model in response to
the output of the loss function. Adam combines the advantages of two
extensions of two stochastic gradient descent algorithms namely
adaptive gradient algorithm(ADA GRAD) and root mean square
propagation(RMSProp).



### 4.5 LIBRARIES USED

#### 4.5.1. OpenCV
OpenCV (Open Source Computer Vision Library) is released under a
BSD license and hence it’s free for both academic and commercial
use. It has C++, Python and Java interfaces and supports Windows,
Linux, Mac OS, iOS and Android. OpenCV was designed for
computational efficiency and with a strong focus on real-time
applications. Written in optimized C/C++, the library can take
advantage of multi-core processing. Enabled with OpenCL, it can
take advantage of the hardware acceleration of the underlying
heterogeneous compute platform.<br />
Adopted all around the world, OpenCV has more than 47 thousand
people of user community and estimated number of downloads
exceeding 14 million. Usage ranges from interactive art, to mines
inspection, stitching maps on the web or through advanced robotics.<br />

#### 4.5.2 Tensorflow
TensorFlow is an open-source software library for dataflow
programming across a range of tasks. It is a symbolic math library,
and is also used for machine learning applications such as neural
networks. It is used for both research and production at Google.
TensorFlow was developed by the Google brain team for internal
Google use. It was released under the Apache 2.0 open source library
on November 9, 2015.<br />
TensorFlow is Google Brain's second-generation system. Version
1.0.0 was released on February 11, 2017. While the reference
implementation runs on single devices, TensorFlow can run on
multiple CPUs and GPUs (with optional CUDA and SYCL extensions
for general-purpose computing on graphics processing units).

TensorFlow is available on 64-bit Linux, macOS, Windows, and
mobile computing platforms including Android and iOS.
Its flexible architecture allows for the easy deployment of computation
across a variety of platforms (CPUs, GPUs, TPUs), and from desktops
to clusters of servers to mobile and edge devices.

#### 4.5.3 Keras
Keras is one of the leading high-level neural networks APIs. It is
written in Python and supports multiple back-end neural
network computation engines.<br />
Given that the TensorFlow project has adopted Keras as the highlevel
API for the upcoming TensorFlow 2.0 release, Keras looks to be
a winner, if not necessarily the winner. In
The biggest reasons to use Keras stem from its guiding principles,
primarily the one about being user friendly. Beyond ease of learning
and ease of model building, Keras offers the advantages of broad
adoption, support for a wide range of production deployment options,
integration with at least five back-end engines (TensorFlow, CNTK,
Theano, MXNet, and PlaidML), and strong support for multiple GPUs
and distributed training.<br />

#### 4.5.4 Numpy
NumPy is a Python library used for working with arrays.It also has
functions for working in domain of linear algebra, fourier transform,
and matrices.NumPy was created in 2005 by Travis Oliphant. It is an
open source project and you can use it freely.NumPy stands for
Numerical Python.<br />
In Python we have lists that serve the purpose of arrays, but they are
slow to process.NumPy aims to provide an array object that is up to
50x faster than traditional Python lists.The array object in NumPy is
called ndarray, it provides a lot of supporting functions that make
working with ndarray very easy.Arrays are very frequently used in
data science, where speed and resources are very important.
NumPy arrays are stored at one continuous place in memory unlike
lists, so processes can access and manipulate them very

efficiently.This behavior is called locality of reference in computer
science.This is the main reason why NumPy is faster than lists. Also it
is optimized to work with latest CPU architectures.<br />

#### 4.5.5 Os
The OS module in python provides functions for interacting with the
operating system. OS, comes under Python’s standard utility
modules. This module provides a portable way of using operating
system dependent functionality. The *os* and *os.path* modules
include many functions to interact with the file system.<br />
Python OS module provides the facility to establish the interaction
between the user and the operating system. It offers many useful OS
functions that are used to perform OS-based tasks and get related
information about operating system.The OS comes under Python's
standard utility modules. This module offers a portable way of using
operating system dependent functionality.The Python OS module lets
us work with the files and directories.<br />

#### 4.5.6 Hunspell (Autocorrect feature)
The python library Hunspell_suggest is used to suggest correct
alternatives for each (incorrect) input word and we display a set of
words matching the current word in which the user can select a word
to append it to the current sentence.This helps in reducing mistakes
committed in spellings and assists in predicting complex words.<br />

#### 4.5.7 Tkinter
Tkinter is the standard GUI library for Python. Python when combined
with Tkinter provides a fast and easy way to create GUI applications.
Tkinter provides a powerful object-oriented interface to the Tk GUI
toolkit.<br />
Creating a GUI application using Tkinter is an easy task. All you need to
do is perform the following steps −<br />
* Import the Tkinter module.<br />
* Create the GUI application main window.<br />

* Add one or more of the above-mentioned widgets to the GUI
application.<br />
* Enter the main event loop to take action against each event
triggered by the user.<br />

#### 4.5.8 PIL (Python Imaging Library)
Python Imaging Library (abbreviated as PIL) (in newer versions known
as Pillow) is a free and open-source additional library for the Python
programming language that adds support for opening, manipulating,
and saving many different image file formats. It is available for
Windows, Mac OS X and Linux. The latest version of PIL is 1.1.7, was
released in September 2009 and supports Python 1.5.2–2.7, with
Python 3 support to be released "later".<br />
Development appears to be discontinued, with the last commit to the
PIL repository coming in 2011. Consequently, a successor project
called Pillow has forked the PIL repository and added Python 3.x
support. This fork has been adopted as a replacement for the original
PIL in Linux distributions including Debian and Ubuntu (since 13.04).<br />


### 4.7 CHALLENGES FACED
There were many challenges faced by us during the making of this
project:<br />
* The very first issue we faced was of dataset. We wanted to deal with
raw images and that too square images as CNN in Keras as it was a
lot more convenient working with only square images. We couldn’t find
any existing dataset for that hence we decided to make our own
dataset.<br />
* Second issue was to select a filter which we could apply on our
images so that proper features of the images could be obtained and
hence then we could provided that image as input for CNN model. We
tried various filter including binary threshold, canny edge detection,
gaussian blur etc. but finally we settled with gaussian blur filter.<br />
* More issues were faced relating to the accuracy of the model we
trained in earlier phases which we eventually improved by increasing
the input image size and also by improving the dataset.<br />

  

