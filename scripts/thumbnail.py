import os
import csv

f = open('D:/AMIT/Dataset/VRT/Image IDs/train-images-boxable-with-rotation.csv', encoding='utf-8')
lines = f.read().split("\n") # "\r\n" if needed

fout = open('thumburls.csv','w', encoding='utf-8')


for l in  csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
	if len(l) > 10 and l[10] != '':
		if len(l) > 11:
			fout.write('%s,%s,%s\n' % (l[0],l[10],l[11]))
		else:
			fout.write('%s,%s,0.0\n' % (l[0],l[10]))

fout.close()
