## Assignment 1
## CMPT 825 Natural Language Processing
* Student Name: Anmol Sharma
* Student ID: asa224
* Student Email ID: anmol_sharma@sfu.ca

## Modelling Word Segmentation problem in Chinese Text as a Sequence Learning Problem suitable for LSTMs.

The following text summarizes the experiments that I performed as I modelled the problem in two different ways, and also discusses the results that were obtained from those implementations. For easy viewing of the code along with its output, an HTML version of the iPython Notebook is added to this directory by the name `Seq2Seq_Prediction.html`.

The modelling was inspired by a variety of papers, but most notably [1]. For this implementation, I use the 1M Chinese Setences Dataset provided by Prof. Sarkar for this assignment purposes.

*__DISCLAIMER:__* The ideas for framing the problem statement were inspired by number of readings, many from my previous experience in machine learning, and some from my current literature review. However the code/implementation is entirely my own. 

__References: [1] Yushi Yao and (2016). Bi-directional LSTM Recurrent Neural Network for Chinese Word Segmentation. CoRR, abs/1602.04874__

## Problem Statements

I rephrased the problem of word segmentation in two ways:

1. [Prob_Def_1] Character-to-Character prediction problem <br>
Where given a single character, predict its corresponding label. Labels in this instance are chosen as - B, M, E, and S which stands for Beginning, Middle, End, and Single-letter-word. The problem could also be solved using a simple neural network model, without RNNs, since its basically just predicting a single output given single input without any context. The major downside of this problem was that it did not leverage the power of RNNs (time dimension) where it utilizes the time dimensions to make decision. However it may be argued that neural network has an internal representation that sort of encodes similar information in this case. This problem was however implemented using an LSTM unit based RNN in Python/Keras.

2. [Prob_Def_2] Sequence-to-Sequence prediction problem <br>
Where given a sequence, predict a sequence of labels. This rephrasing of problem allows me to utilize the time dimension property of RNN where it sees the context of each character while making the decision predicting the corresponding label. This problem was implemented using LSTM unit based RNN in Python/Keras.

**Now I provide an overview of the various data preprocessing steps common to both of these problem definitions that were performed in order bring the dataset close to what I could for training a neural network model.**

### Parsing and Assigning Labels
Given a set of words, we assign the labels to each character seen in the training example. For example:

`A GOOD MAN ANMOL SHARMA` <br>
`| |||| ||| ||||| ||||||` <br>
`S BMME BME BMMME BMMMME`

The labels were converted to categorical vector representation as well, using this truth table -

~~~~
      | Integer | Vector
Label | Value   | Representation
--------------------------------
B     | 0       | 1,0,0,0
M     | 1       | 0,1,0,0
E     | 2       | 0,0,1,0
S     | 3       | 0,0,0,1
~~~~

This parsing produces a list of tuples of the form `[(u'\ue12as', 0), (u'\ue4a3', 2)....(u'\u2354', 3)]` where the first element of the tuple is a character and the second element is the class label. It looks like this:

![Nothing](ims/sequence.png)

For [Prob_Def_1], newline characters were not parsed and were skipped. It was assumed that the input test file will have newlines to determine where to end a line which the RNN was working on.

However, for [Prob_Def_1] the issue arises where the newline character either have to be 1) ignored, or 2) incorporated in the parsing, probably as a standalone character. After some initial experiments, I found that ignoring newline would lead me to lose information about lines, which the network cannot determine at test time. So the network lead to generating an output text with basically no new lines. After some literature review, I found out that actually parsing newline as a single-letter-character can help the RNN make decisions, since context often changes after each line, and newline character can help RNN to make that decision.

Hence for [Prob_Def_2] we parse newline characters as having label "S". One interesting implementation related thing is that the newline chracter is encoded as a special symbol in unicode, since I was having issues reading it back from a file. The special symbol is chosen at random,and appears to be a degree sign.

### Initial Integer Embedding

So machine learning models cannot directly work on categorical data, the issue arises that the data must converted to an integral representation to make it compatible for training. The method that I use for this is simple, and is highlighted here using English language -

Given an English alphabet of unique characters, we find there are 26 unique characters. Hence, for each character, we assign an integer value such that:

`A B ...... X   Y   Z`<br>
`| | ...... |   |   |`<br>
`0 1 ...... 23 24 25`<br>

I implement this for chinese characters by finding the number of unique characters seen in the training set, which totalled to about 5920. A pair of dictionaries were generated (one for word-to-int and other for int-to-word) for converting the dataset.

### Chunking into Sequences

For [Prob_Def_1], we used the following approach:

1. Generate each sentence in the training data as a single sequence.
2. Use a moderate value for fixing the sequence size to, in my case I used `maxlen = 200. `
3. For each `sequence)`:
  1. if len(sequence) < maxlen:
    1. Pad the sequence until `len(sequence) == maxlen`
  2. else:
    1. Truncate the sequence from front so that `len(sequence) == maxlen`

This approach was necessary since RNNs expect inputs to have a set size.

However, for [Prob_Def_2] I experimented with constant size sequences, where I create each sequence of `maxlen = 13` along with parsing the newline character. This value was used after reading the paper [1]

## Training Data

After the preprocessing is done, the training set is generated with shapes as follows:

![Nothing](ims/data_shape.png)

## Defining RNN Model

The model that I train for [Prob_Def_1] is as follows:

~~~~
model = Sequential()
# first argument is the size of the vocabulary, second argument is the size of embedding, third argument is the
# number of features in the text, we only have 1 character.
model.add(Embedding(len(orig_dict), 200, input_length=1))
model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Bidirectional(LSTM(10)))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
~~~~

The model that I train for [Prob_Def_2] is as follows:
~~~~
model = Sequential()
model.add(Embedding(input_dim=len(orig_dict), output_dim=200, input_length=n_timesteps)) # dictionary size, embedding vector size, timesteps
model.add(LSTM(300, return_sequences=True))
model.add(LSTM(300, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(4, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
model.summary()
~~~~

The model for the above code has the following output shapes per each layer:

![Nothing](ims/model_seq2seq.png)

### Embedding Layer
`model.add(Embedding(input_dim=len(orig_dict), output_dim=200,`<br>
An embedding layer was used as the input layer to our RNN which learns the most effective emdedding of the words using a set of weights by projecting each character onto a d-dimensional space and finding the most suitable internal representation (embedding) that maximized the goal of RNN (in this case, produce accurate predictions). The embedding maxlen was set to `output_dim = 200`, which is the same value used in paper [1]. input_dim parameter takes the number of unique characters in the training corpus.

### LSTM Layer

`model.add(LSTM(300, return_sequences=True))`

Long-Short Term Memory Networks is an extension of Recurrent Neural Networks which addresses two main problems plaguing traditional RNNs:

1. Vanishing/Exploding gradient problem.
2. Retaining long-term memory, which is sometimes required for modelling data that has long-term dependencies, like text.

LSTM unit generally looks like this:

![Nothing](ims/lstm_unit.png)

(Figure courtesy - http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

It has usually three gates -

1. Input Gate
2. Output Gate
3. Forget Gate

In my implementation, I use an LSTM layer with 300 LSTM units. Each LSTM layer usually outputs only the predicted sequence, however in my case where I wanted to stack another LSTM on top of it, the layer returns back input sequences along with its predictions which are then passed on to the next LSTM layer.

### Dropout Layer

`model.add(Dropout(0.2))`

This layer turns off the input neurons randomly to address the issue of overtraining. Usually the layer generates a random number for each input neuron (or activation), and if the number is greater than the given probability (here `p = 0.2`) the neuron is turned off. This allows the network to learn features that are not highly correlated with other features.

### Time Distributed Dense Layer

`model.add(TimeDistributed(Dense(4, activation='softmax')))`

The TimeDistributed wrapper in Keras allows the network to apply the Dense layer (which has the same number of neurons as the output classes, which in our case `n_output = 4`) to each timestamp (`n_timesteps=13` in our case) with the same weights. This is useful in problems which are inherently Many-to-Many Sequence modelling problem, which our [Prob_Def_2] essentially is.

## Train the Model

~~~~
model.fit(X_train, y_train, epochs=10,
          batch_size=100, verbose=1, validation_data=[X_test, y_test], callbacks=[mc])
~~~~
The model was trained using `Adadelta` optimizer with default values for learning rate (`lr = 1.0`), rho (`rho = 0.95`) and epsilon (`eps = 1e-8`). Adadelta optimizer is designed in a way that makes the initial choise of learning rate less determining of the fact that the network converges to a good optimal or not. It does this by automatically throttling learning rate using the current values of gradients that propagates through the network.

A single epoch of training gives good results on both training and test set, as shown below:

![Nothing](ims/training.png)

## Discussion and Conclusion
The accuracy of the model goes upto 90% on test set, and around 91% on the training set after about 10 epochs of training. Each epoch takes approximately 800s to complete.

However, the accuracy on test set during this phase is not entirely reflected on the test data that Prof. Sarkar provided. The test set accuracy for the given test input was about 75% for this model. After more in-depth analysis and review of the issues, I concluded the following:

1. The problem definition with 4 different labels makes it hard to take decision during the test time. This can be explained using a simple example:

Let the test sequence be:
`ANMOLSHARMA` <br>

and the predicted sequence be:
`BEMMEBMMMMS`

Out of 10 values the network "correctly" predicted 8 values, but incorrectly predicted 2 values. This makes the accuracy of the network 80%. However, the incorrect predictions directly impact the segmentation of the word. The question arises, how do we segment the word?

`AN MOL SHARM A`

or

`AN MOL SHARMA`

The approach that I applied suffers due to these ambiguities, and explicit rules have to be written during post-processing to handle such cases.

*__However, such rules are subjective. Hence the approach performed poorly than a unigram segmenter. __*

Unfortunately I figured this out AFTER doing all this, but nonetheless it was a great exercise, and I learnt a great deal about RNNs, LSTMs, BLSTM, and how to implement actual NLP solutions.
