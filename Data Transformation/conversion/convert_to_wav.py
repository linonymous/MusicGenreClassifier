import os
import sys


path = "C:\\Users\\Swapnil.Walke\\Downloads\\genres.tar\\genres\\"
lis = os.listdir(path + "\\blues")
genres = os.listdir("C:\\Users\\Swapnil.Walke\\Downloads\\genres.tar\\genres\\")
for genre in genres:
    src_path = path + "\\" + genre + "\\"
    dest_path = path + "\\" + genre + "_wav" + "\\"
    os.mkdir(dest_path)
    files = os.listdir(src_path)
    for file in files:
        src_file = src_path + file
        dest_file = dest_path + file[:-3] + ".wav"
        cmd = "sox " + src_file + " -e signed-integer " + dest_file
        os.system(cmd)

