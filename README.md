# ANUVADAK-Sign-language-to-text-convertor
## CONTENTS 
<br>[Abstract ](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#abstract)
<br>[Chapter 1 Introduction](https://github.com/Arjun-Narula/Anuvadak-Sign-language-to-text-convertor/blob/main/README.md#chapter-1-introduction)
<br>Chapter-2-Literature Survey 
<br>Chapter-3-Artificial Neural Network – A Review 
<br>Chapter-4-Methodology 
<br>Chapter-5-Results and Applications 
<br>Chapter-6-Conclusions and Future Scope 
<br>References 
<br>Appendix

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
  

