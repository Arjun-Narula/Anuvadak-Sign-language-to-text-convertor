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
1 To develop an application interface that interprets American Sign
  Language to Text in real time to assist deaf and dumb for
  communicating with them effectively eliminating the requirement of a
  translating individual.<br />
2 To devise a model to achieve the highest possible accuracy and least
  time consumption for prediction of symbols as compared to already
  existing models.<br />
3 To reduce the cost and develop an economical and user friendly
  graphical user interface (GUI) application that requires minimal
  maintenance for conversion of sign to its corresponding text.<br />
4 To provide suggestions based on current word to eliminate the need
  of translating the full word, thereby improving accuracy and reducing
  time for sign to text conversion.<br />
5 To reduce the chances of spelling mistakes by suggesting correct
  spellings from English dictionary words close to the current word.
6 To formulate characters, words and sentences with interpretation of
  symbols in American Sign Language in a portable and real time
  application.<br />

