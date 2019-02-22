import pickle
import numpy as np
import re

# Path to the .txt files
attributes_dir = ''

# Path to save pickle files
pickle_save_dir = ''

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

Process the list_attr_cloth and place the processed data in appropriate data structure
"""

print('Proceeding list_attr_cloth.txt')
file = open("list_attr_cloth.txt", 'r')

attributeNames = []
attributeTypes = {}

fileSize = int(file.readline())
file.readline()

for _ in range(fileSize):
    newLine = file.readline()
    splitList = re.split(r'[ ]{2,}', newLine)
    attributeNames.append(splitList[0])
    attributeTypes[attributeNames[-1]] = splitList[1].split('\n')[0]
file.close()

attributeTypes[attributeNames[446]] = '5'

"""
File format
289222
image_name  evaluation_status
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                        train
img/Sheer_Pleated-Front_Blouse/img_00000002.jpg                        train
img/Sheer_Pleated-Front_Blouse/img_00000003.jpg                        val
img/Sheer_Pleated-Front_Blouse/img_00000004.jpg                        train
img/Sheer_Pleated-Front_Blouse/img_00000005.jpg                        test

Process the list_eval_partition and place the processed data in appropriate data structure
"""

print('Proceeding list_eval_partition.txt')
file = open(attributes_dir + "list_eval_partition.txt", 'r')

evalImageNames = {}

fileSize = int(file.readline())
file.readline()

for _ in range(fileSize):
    newLine = file.readline()
    splitList = re.split(r'\s', newLine)
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

Process the list_attr_img and place the processed data in appropriate data structure
"""

print('Proceeding list_attr_img.txt')
file = open(attributes_dir + "list_attr_img.txt", 'r')
imageNames = []
attributeLabels = []

fileSize = int(file.readline())
file.readline()

for i in range(fileSize):
    newLine = file.readline()
    splitList = re.split(r'[\s]{1,}', newLine)
    filterredList = list(filter(None, splitList))
    imageNames.append(filterredList[0])
    filterredList = [number.replace('-1', '0') for number in filterredList[1:]]
    convertedList = list(map(int, filterredList))
    attributeLabels.append(convertedList)

file.close()

attributeLabels = np.array(attributeLabels)

# Couple each partition to a number
attributesCate = [['1', 'Texture'], ['2', 'Fabric'], ['3', 'Shape'], ['4', 'Part'], ['5', 'Style']]

# DataSets:
# Process each partition and split the processed partition into train, test, and val. Each of these will contain
# a set of images and labels. Then they re stored in pickle file for use.
# Attributes:
# The attributes for each partition will be stored into a list and saved into a pickle file
print('Starting list_attr_cloth.txt')
for num in attributesCate:
    print('Start Processing '+num[1])

    Dataset = {'train': {'names': [],
                         'labels': []},
               'val': {'names': [],
                       'labels': []},
               'test': {'names': [],
                        'labels': []}
               }
    attributelist = []
    indices = []

    for s in range(1000):
        if attributeTypes[attributeNames[s]] == num[0]:
            attributelist.append(attributeNames[s])
            indices.append(s)

    names = []
    for i in range(len(imageNames)):
        for j in indices:
            if attributeLabels[i][j] == 1:
                Dataset[evalImageNames[imageNames[i]]]['names'].append(imageNames[i])
                Dataset[evalImageNames[imageNames[i]]]['labels'].append([])
                names.append(imageNames[i])
                break;

    counters = {"train": 0,
                "val": 0,
                "test": 0}

    for i in range(len(imageNames)):
        if imageNames[i] in names:
            for j in indices:
                Dataset[evalImageNames[imageNames[i]]]['labels'][counters[evalImageNames[imageNames[i]]]].append(
                    attributeLabels[i][j])
            counters[evalImageNames[imageNames[i]]] += 1

    Dataset['train']['labels'] = np.array(Dataset['train']['labels'])
    Dataset['val']['labels'] = np.array(Dataset['val']['labels'])
    Dataset['test']['labels'] = np.array(Dataset['test']['labels'])

    print(num[1]+' data set saved as pickle')
    pickle_out = open(pickle_save_dir + num[1] + "DataSet.pickle", "wb")
    pickle.dump(Dataset, pickle_out)
    pickle_out.close()

    print(num[1]+' attributes saved as pickle')
    pickle_out = open(pickle_save_dir + num[1] + "Attributes.pickle", "wb")
    pickle.dump(attributelist, pickle_out)
    pickle_out.close()
    print('Finish Processing '+num[1])

