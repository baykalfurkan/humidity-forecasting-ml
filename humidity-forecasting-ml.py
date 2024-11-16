# -*- coding: utf-8 -*-

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme
#2.1. Veri Yukleme
veriler = pd.read_csv('tenis.csv')


#veri on isleme
#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1] #en sol sütun seçilir
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)



#backward elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1) #Bu satırda, numpy kütüphanesinin np.append fonksiyonu kullanılarak veri adlı bir veri setine, tüm elemanları 1 olan bir sütun ekleniyor
'''
np.ones((22,1)).astype(int): Bu kısım, tüm elemanları 1 olan 22 satır ve 1 sütundan oluşan bir numpy dizisi oluşturur. astype(int) metodu ile elemanlar tam sayı (integer) olarak ayarlanır. Bu sütun, veri dizisinin başına eklenecek.

np.append(arr=..., values=..., axis=1): Bu fonksiyon, veri dizisine yeni bir sütun eklemek için kullanılır.

arr=np.ones((22,1)).astype(int): İlk parametre arr, eklenmek istenen sütunun olduğu diziyi ifade eder (burada tüm elemanları 1 olan sütun).
values=veri: İkinci parametre values, veri dizisini belirtir. Bu dizi, 1’ler sütununun sağ tarafına eklenecek.
axis=1: Bu argüman, eklemenin yatay olarak (sütun olarak) yapılacağını belirtir.

çoklu doğrusal regerasyona benzetmek için yapılır.
'''


X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())
#en yüksek p value çıkarılır

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())


x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
#windy sütunu silindi

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


y_pred = y_pred.ravel() 
y_test = y_test['humidity'].values 


from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(30*"-"+"\nDeger - Tahmin edilen deger")
for i, (pred, test) in enumerate(zip(y_pred, y_test), start=1):
    print(f" {pred:.2f} -  {test}")

print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"R² Skoru: {r2:.2f}")
print(30*"-")

from sklearn.metrics import mean_absolute_error, r2_score


