import sys
import random
import pandas as pd

from datetime import datetime
from datetime import timedelta

import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_excel('C:/Users/USER/NoteBooks/Dropex-Wardenine-Final_fake.xlsx')
df['delta_days'] = (df['Date arrivée'] - df['Date']).dt.total_seconds() / (60 * 60 * 24)

enc_gov = LabelEncoder()
df['Governorate_nb'] = enc_gov.fit_transform(df['Governorate'])
df['original_delta_days'] = df['delta_days']
X = df[['COD', 'Governorate_nb']]
enc_days = LabelEncoder()
df['delta_days'] = enc_days.fit_transform(df['delta_days'])

y = df['delta_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
df
RFC = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=51)
RFC.fit(X_train, y_train)
pred = RFC.predict(X_test)
acc_rfc=accuracy_score(y_test, pred)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize=True)
# Supervised Learning Estimators

# Support Vector Machines (SVM)
from sklearn.svm import SVC

svc = SVC(kernel='linear')

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

# KNN
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# Unsupervised Learning Estimators

# Principal Component Analysis (PCA)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)

# K Means

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=3, random_state=0)
# Model Fitting | Supervised learning

lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
gnb.fit(X_train, y_train)

# Model Fitting | Unsupervised learning

k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)
y_pred_svc = svc.predict(X_test)
y_pred_gnb = gnb.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_knn = knn.predict_proba(X_test)
y_pred = k_means.predict(X_test)
accuracy_score(y_test, y_pred)

# val = input("Enter your Governorate : ")
#
# options = {"Ariana" :0 ,
#            "Beja" :1 ,
#            "Ben Arous" :2 ,
#            "Bizerte" :3 ,
#            "Gabes" :4 ,
#            "Gafsa" :5 ,
#            "Jendouba" :6 ,
#            "Kairouan" :7 ,
#            "Tunis" :22 ,
#            "Zaghouan" :23 ,
#            "Kasserine" :8 ,
#            "Kébili" :9 ,
#            "Le Kef" :11 ,
#            "Mahdia" :12,
#            "La Manouba" :10 ,
#            "Médenine" :14 ,
#            "Monastir" :13 ,
#            "Nabeul" :15 ,
#            "Sfax" :16 ,
#           "Sidi Bouzid" :17,
#            "Siliana" :18 ,
#            "Sousse" :19 ,
#            "Tataouine" :20 ,
#            "Tozeur" :21 }
#
# gov=options[val]
# val_cod = input("Enter your order's price : ")
# test=RFC.predict([ [val_cod,gov] ])
# print(int(test[0]))
#
# value = random.uniform(0,4)
# print(value)
# print (datetime.now() + timedelta(days=int(test[0])+1,hours=value))

import wx

class windowClass(wx.Frame):
    def __init__(self,*args,**kwargs):
        super(windowClass,self).__init__(*args,**kwargs,size=(800,600))
        self.basicGUI()
    def basicGUI(self):
        panel=wx.Panel(self)

        menuBar=wx.MenuBar()
        fileButton=wx.Menu()
        exitItem=wx.MenuItem(fileButton,wx.ID_EXIT,'Exit')
        exitItem.SetBitmap(wx.Bitmap('C:/Users/USER/NoteBooks/1.png'))
        fileButton.Append(exitItem)

        menuBar.Append(fileButton,'File')
        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_MENU,self.Quit,exitItem)
        yesNoBox=wx.MessageDialog(None,'We can help you predict when your order will come','Welcome',wx.YES_DEFAULT)
        yesNoAnswer = yesNoBox.ShowModal()
        yesNoBox.Destroy()
        chooseOneBox = wx.SingleChoiceDialog(None,'Choose your Governorate','Governorate choice',["Ariana" ,"Beja","Ben Arous","Bizerte" ,"Gabes" ,"Gafsa" ,"Jendouba","Kairouan" ,"Tunis" ,"Zaghouan" ,"Kasserine" ,"Kébili" ,"Le Kef" ,"Mahdia","La Manouba","Médenine","Monastir","Nabeul","Sfax","Sidi Bouzid","Siliana" ,"Sousse" ,"Tataouine","Tozeur"])
        if chooseOneBox.ShowModal()==wx.ID_OK:
            val=chooseOneBox.GetStringSelection()
        nameBox=wx.TextEntryDialog(None,"Enter your order's price","Order's price")
        if nameBox.ShowModal()==wx.ID_OK:
            val_cod=nameBox.GetValue()
        options = {"Ariana": 0,
                              "Beja" :1 ,
                              "Ben Arous" :2 ,
                              "Bizerte" :3 ,
                              "Gabes" :4 ,
                              "Gafsa" :5 ,
                              "Jendouba" :6 ,
                              "Kairouan" :7 ,
                              "Tunis" :22 ,
                              "Zaghouan" :23 ,
                              "Kasserine" :8 ,
                              "Kébili" :9 ,
                              "Le Kef" :11 ,
                              "Mahdia" :12,
                              "La Manouba" :10 ,
                              "Médenine" :14 ,
                              "Monastir" :13 ,
                              "Nabeul" :15 ,
                              "Sfax" :16 ,
                             "Sidi Bouzid" :17,
                              "Siliana" :18 ,
                              "Sousse" :19 ,
                              "Tataouine" :20 ,
                              "Tozeur" :21 }

        gov=options[val]
        test=RFC.predict([ [val_cod,gov] ])
        print(int(test[0]))

        value = random.uniform(0,4)
        print(value)
        res=datetime.now() + timedelta(days=int(test[0])+1,hours=value)
        print (res)
        acc_str=str(acc_rfc*100)+"%"
        final_res="Your order will be delivered on \n"+str(res)+"\nAccuracy:"+" "+acc_str


        aweText=wx.StaticText(panel,-1,final_res,(3,3))
        aweText.SetForegroundColour('#67cddc')
        aweText.SetBackgroundColour('black')
        font = wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.BOLD)
        aweText.SetFont(font)
        png = wx.Image("C:/Users/USER/NoteBooks/2.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        wx.StaticBitmap(panel, -1, png, (200, 100), (png.GetWidth(), png.GetHeight()))
        self.SetTitle('Predicting Delivery Date & Time')
        self.Show(True)
    def Quit(self,e):
        self.Close()
def main():
    app=wx.App()
    windowClass(None)
    app.MainLoop()
main()