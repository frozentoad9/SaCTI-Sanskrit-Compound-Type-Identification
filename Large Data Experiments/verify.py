import pandas as pd
from sklearn.model_selection import train_test_split
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import matplotlib.pyplot as plt
test = pd.read_csv('test_large.csv')#.drop(["Unnamed: 0"],axis=1)
dev = pd.read_csv('dev_large.csv')#.drop(["Unnamed: 0"],axis=1)
train = pd.read_csv('train_large.csv')#.drop(["Unnamed: 0"],axis=1)

def check_num_samples1(train,dev,test):
    print("samples shape in train ",train.shape)
    print("samples shape in dev ",dev.shape)
    print("samples shape in test ", test.shape)
    print()
    print("total len of Colling in train data",len(train["Compounds"]))
    print("unique entries in Colling train data",len(set(train["Compounds"])))
    print("total len of Colling in test data",len(test["Compounds"]))
    print("unique entries in Colling test data",len(set(test["Compounds"])))
    #print("finding intersection bw train and test set ",set(test["Compounds"]).intersection(set(train["Compounds"])))

    print("********************************************************************************")
    print("total len of Components in train data",len(train["Context"]))
    print("unique entries in Components train data",len(set(train["Context"])))
    print("total len of Components in test data",len(test["Context"]))
    print("unique entries in Components test data",len(set(test["Context"])))
    print("finding intersection bw train and test set ",len(set(test["Context"]).intersection(set(train["Context"]))))
    print("**********************************************************************************")
    print("unique entries in Final_Clean_Context_d test data",len(set(test["Context"])))
    print("finding intersection bw train and test set ",len(pd.merge(train, test, how ='inner'))) #becz 

    print("***********************************************************************************")
    print("***********************************************************************************")

    print("total len of Colling in train data",len(dev["Context"]))
    print("unique entries in Colling train data",len(set(dev["Context"])))
    print("finding intersection bw dev and test set ",len(set(dev["Context"]).intersection(set(test["Context"]))))

    print("********************************************************************************")
    print("total len of Components in dev data",len(dev["Context"]))
    print("unique entries in Components dev data",len(set(dev["Context"])))
    print("finding intersection bw dev and test set Components column",len(set(dev["Context"]).intersection(set(test["Context"]))))
    print("finding intersection bw dev and test set ",len(pd.merge(dev, test, how ='inner'))) #becz 

check_num_samples1(train,dev,test)

#2.check ratio bw all 4 classes
# def check_ratios(train,test):
l1 = train["Tag"].value_counts()
l2 = dev['Tag'].value_counts()
l1,l2 = dict(l1),dict(l2)
print(l1,l2) 
vals1 = list(l1.values())
vals2 = list(l2.values())
rat1 = vals2/vals2[-1]
rat2 = vals1/vals1[-1]

ratios = [i/j for i,j in zip(rat1,rat2)]
print()
print()
print("    ",l1.keys())
print("ratios classes ",ratios)

print("for dev set")
print(len(vals2),"   ",len(set(dev["Tag"])))
print(len(set(dev["Tag"])-set(train["Tag"])))
print(set(dev["Tag"])-set(train["Tag"]))


print("for test set")
print(len(set(test["Tag"])-set(train["Tag"])))
print(set(test["Tag"])-set(train["Tag"]))

cp = []
for i in train["Tag"].to_list():
    if i not in test['Tag'].to_list():
        cp.append(i)
print("not presenet entries tag with test",len(cp))
    
cp = []
for i in train["Tag"].to_list():
    if i not in dev['Tag'].to_list():
        cp.append(i)
print("not presenet entries dev",len(cp))
    



#For cooling Data
# check_ratios(train,test)
#Please calculate accordingly as per ratios needed
