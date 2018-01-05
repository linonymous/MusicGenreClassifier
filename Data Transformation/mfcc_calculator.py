import sys
sys.path.append("C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\pymir")
from pymir import AudioFile
wavData = AudioFile.open("C:\\Users\\Swapnil.Walke\\Downloads\\genres.tar\\genres_wav\\blues_wav\\blues.00000.wav")
fixedFrames = wavData.frames(1024)
spectra = [f.spectrum() for f in fixedFrames]
a = spectra[0].mfcc2(13)
print "MFCC calculations"
print a
print "**************************************************"
b= spectra[0].rolloff()
print "spectral roll off"
print b
c = spectra[0].chroma()
print "Chroma frequency"
print c
print len(c)
d = spectra[0].centroid()
print "Spectral Centroid"
print d
"""done with following 5 characteristics, but need to look upon how to calculate them all for one whole file
- mfcc
- spectral centroid
- spectral roll off
- chroma frequency
- zero crossing rate"""