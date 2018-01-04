import sys
sys.path.append("C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\pymir")
from pymir import AudioFile
wavData = AudioFile.open("C:\\Users\\Swapnil.Walke\\Downloads\\genres.tar\\genres_wav\\blues_wav\\blues.00000.wav")
fixedFrames = wavData.frames(1024)
spectra = [f.spectrum() for f in fixedFrames]
a = spectra[0].mfcc2(13)
print a
