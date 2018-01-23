# MusicGenreClassifier
   
   This is the implementation of the Music Genre Classifier given in this [paper](https://cse.iitk.ac.in/users/cs365/2015/_submissions/archit/report.pdf)

- ****Dataset****                   
                                               
    The GTZAN data set used is taken from this [website](http://marsyasweb.appspot.com/download/data_sets/). This includes following genres:                                                                                                                                    
    - Blues                        
    - Classical                  
    - Country                    
    - Disco                       
    - Pop          
    - Jazz               
    - Reggae            
    - Rock              
    - Metal
    
    Each audio clip in the data set has a length of 30 seconds, are 22050Hz Mono 16-bit MPEG files.
    
- ****Features****

    Following features has been used to do this classification : 
    
    - Mel Frequency Cepstral Coeficient
    - Spectral Centroid
    - Spectral Rolloff
    - Chroma Frequency
    - Zero crossing rate

    All of the above Features, together contribute to a feature vector with 28 different features.

- ****Classifier****

    Following classifiers are being trained :

    - Logistic regression
    - SVM

    Accuracy of datset being classified is quiet low, for 10 classes, and increases with reduction in  number of classes
    for prediction. Analyzing different curves to understand what would increase accuracy. First suspicion is low number of
    dataset examples.
