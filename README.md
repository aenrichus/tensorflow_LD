# tensorflow_LD

This project is an attempt to replicate a paper that used convolutional neural networks to replicate the performance of baboons in a lexical decision task [1]. The network takes an image of a string of characters as input and makes a binary decision about whether the string constitutes a word. Also included is the code used to generate and load images of text from a list of words.

Differences between this model and that used in the paper include:

1. Input images use different fonts and colors, and may be a different size.

2. The convolutional filter is of size 3x3 instead of a larger filter.

3. There are three convolutional and pooling layers instead of two.

4. Other parameters, such as the learning rate, may also differ.

This was used as an example during the "Deep Learning 101" presentation that was given on 11 January 2017 at the University of Connecticut. The talk was an introduction to machine learning and deep learning in Tensorflow for professors and graduate students in the Neurobiology of Language program. Participants came from 7 Ph.D. programs (incl. Linguistics; Speech, Language, and Hearing Sciences; Physiology & Neurobiology; and 4 programs in Psychology: Behavioral Neuroscience, Clinical Psychology, Developmental Psychology and Perception-Action-Cognition) and generally did not have a significant background in computer science.

## References
[1] Hannagan, Thomas, Johannes C. Ziegler, Stéphane Dufau, Joël Fagot, and Jonathan Grainger. "Deep learning of orthographic representations in baboons." PloS one 9, no. 1 (2014): e84843. doi:10.1371/journal.pone.0084843
