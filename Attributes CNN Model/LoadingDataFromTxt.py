import pickle
import re
import numpy as np
"""
File format
1000
attribute_name  attribute_type
a-line                       3
abstract                     1
abstract chevron             1
abstract chevron print       1
abstract diamond             1
abstract floral              1
"""
file = open("list_attr_cloth.txt", 'r')

attributeNames = []
attributes = {}

fileSize = int(file.readline())
file.readline()

for _ in range(fileSize):
    newLine = file.readline()
    splitList = re.split(r'[ ]{2,}',newLine)
    attributeNames.append(splitList[0])
    attributes[attributeNames[-1]] = splitList[1].split('\n')[0]
file.close()

pickle_out = open("AttributesNames.pickle","wb")
pickle.dump(attributeNames, pickle_out)
pickle_out.close()

pickle_out = open("AttributesTypes.pickle","wb")
pickle.dump(attributes, pickle_out)
pickle_out.close()
"""
File format
289222
image_name  evaluation_status
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                        train
img/Sheer_Pleated-Front_Blouse/img_00000002.jpg                        train
img/Sheer_Pleated-Front_Blouse/img_00000003.jpg                        val
img/Sheer_Pleated-Front_Blouse/img_00000004.jpg                        train
img/Sheer_Pleated-Front_Blouse/img_00000005.jpg                        test
"""
file = open("list_eval_partition.txt", 'r')

evalImageNames = {}

fileSize = int(file.readline())
file.readline()

for _ in range(fileSize):
    newLine = file.readline()
    splitList = re.split(r'\s',newLine)
    evalImageNames[splitList[0]] = splitList[-2]
file.close()

"""
File format
289222
image_name  attribute_labels                                           1000 entries 
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                         1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
img/Sheer_Pleated-Front_Blouse/img_00000002.jpg                        -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
....
img/Sheer_Pleated-Front_Blouse/img_00000005.jpg                        -1 .....

the file does not contain a zero -- checked with:

fileSize = int(file.readline())
file.readline()
for i in range(fileSize):
    newLine = file.readline()
    if re.findall(r' 0 ', newLine):  replace the zero with 1 or -1 if you need to check how it works...
        print(i)
file.close()
"""
file = open("list_attr_img.txt", 'r')
imageNames = []
attributeLabels = []

train = []
trainLabels = []

val = []
valLabels = []

test = []
testLabels = []

fileSize = int(file.readline())
file.readline()

for i in range(fileSize):
    newLine = file.readline()
    splitList = re.split(r'[\s]{1,}',newLine)
    filterredList = list(filter(None, splitList))
    imageNames.append(filterredList[0])
    filterredList = [number.replace('-1', '0') for number in filterredList[1:]]
    convertedList = list(map(int, filterredList))
    attributeLabels.append(convertedList)

    if evalImageNames[imageNames[-1]] == 'train':
        train.append(imageNames[-1])
        trainLabels.append(attributeLabels[-1])
    elif evalImageNames[imageNames[-1]] == 'val':
        val.append(imageNames[-1])
        valLabels.append(attributeLabels[-1])
    else:
        test.append(imageNames[-1])
        testLabels.append(attributeLabels[-1])

file.close()

print (np.shape(train))
print (np.shape(val))
print (np.shape(test))

attributeLabels = np.array(attributeLabels)
trainLabels = np.array(trainLabels)
valLabels = np.array(valLabels)
testLabels = np.array(testLabels)

pickle_out = open("TrainLabels.pickle","wb")
pickle.dump(trainLabels, pickle_out)
pickle_out.close()

pickle_out = open("TrainImageNames.pickle","wb")
pickle.dump(train, pickle_out)
pickle_out.close()

pickle_out = open("ValLabels.pickle","wb")
pickle.dump(valLabels, pickle_out)
pickle_out.close()

pickle_out = open("ValImageNames.pickle","wb")
pickle.dump(val, pickle_out)
pickle_out.close()

pickle_out = open("TestLabels.pickle","wb")
pickle.dump(testLabels, pickle_out)
pickle_out.close()

pickle_out = open("TestImageNames.pickle","wb")
pickle.dump(test, pickle_out)
pickle_out.close()