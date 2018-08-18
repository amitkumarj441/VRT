import os
import sys
import os.path

f = open('D:/DATASET/VRT/Annotations/challenge-2018-train-vrd.csv','r', encoding='utf-8')
vrd_lines = f.read().split('\n') # "\r\n" if needed
f.close()

fout = open('annotations.csv','w', encoding='utf-8')

for line in vrd_lines:
    if line != "": # add other needed checks to skip titles
        words = line.split(',')
        img_id = words[0]
        fname = 'D:/AMIT/Dataset/VRT/train/' + img_id + '.jpg'
        if os.path.isfile(fname) == True:
            fout.write('%s\n' % (line))
fout.close()
