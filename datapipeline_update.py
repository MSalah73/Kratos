"""
balancing the training dataset and shorten the preprocessing time; the original pipeline was replaced
DISTRIBUTED UNDER AN MIT LICENSING.
"""
import random
#about five minutes for random shuffle
txt_path = "list_color_cloth.txt"
split_path = "list_eval_partition.txt"
imageset = []
labelset = []
f = open(split_path)
i = 0
split_dic = {}
split_dic["train"] = []
split_dic["test"] = []
while i < 52714:
    line = f.readline()
    if i != 0 and i != 1:
        img_path, temp, update = line.split()
        if update == "train" or update == "gallery":
            split_dic["train"].append(img_path)
            random.shuffle(split_dic["train"],random.random)
        else:
            split_dic["test"].append(img_path)
            random.shuffle(split_dic["test"],random.random)
    i += 1
f.close()
print("split down")
#print(len(split_dic["train"]), len(split_dic["test"]))
imageset = []
labelset = []
classes = []
f = open("list_color_cloth.txt")
i = 0
while i < 52714:
    line = f.readline()
    if i != 0 and i != 1:
        img_path, color = line.split(maxsplit=1)
        classes.append(color)
        imageset.append(img_path)
        labelset.append(color)
    i += 1
f.close()
classes = list(set(classes))#no replicates color label list, all colors the model could predict on
#replace letter labels by int index
temp = []
for label in labelset:
    labelnum = classes.index(label)
    temp.append(labelnum)
labelset = temp#same order numerical label representation
#for i in range(10):
    #print(classes[i])
print("read in label down")
#write to a permanent txt file
#important the order of "allcolor.txt" file cannot be changed!!!orders have been checked
with open('allcolor.txt', 'w') as filehandle:  
    filehandle.writelines("%s" %color for color in classes)
trainlabelset = []
testlabelset = []
#write random order to txt files
with open('traindataset.txt', 'w') as filehandle:  
    filehandle.writelines(" %s\n" %img for img in split_dic["train"])#38494
with open('testdataset.txt', 'w') as filehandle:  
    filehandle.writelines(" %s\n" %img for img in split_dic["test"])#14218
print("time up")
for img in split_dic["train"]:
    trainlabelset.append(labelset[imageset.index(img)])
for img in split_dic["test"]:
    testlabelset.append(labelset[imageset.index(img)])
with open('trainlabelset.txt', 'w') as filehandle:
    filehandle.writelines(" %s\n" %num for num in trainlabelset)#the corordiante index number to the classes list
with open('testlabelset.txt', 'w') as filehandle:
    filehandle.writelines(" %s\n" %num for num in testlabelset)#the corordinate index number to the classes list
