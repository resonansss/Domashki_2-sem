import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pylab as plt
import seaborn as sns



data = pd.read_csv(r"C:\Users\mkuzo\Desktop\Программирование_2 сем\mag2022\CL\term02\01-Intro, LinReg\Car details v3.csv")
print(data.head())
print(data.size)
print(data.shape)
print(data.dtypes)
print(data.dtypes.value_counts())

#год year, цена selling_price и km_driven оставлю без изменений

#поработаем с fuel
#мне кажется, что тип топлива может влиять на цену, поэтому надо вывести все значения и посмотреть, какие они бывают, и попробовать рассклассифицировать их
#print(data['fuel'].unique()) #четыре значения ('Diesel' 'Petrol' 'LPG' 'CNG')

def map_fuel(fuel):
    repl = dict(zip(['Diesel', 'Petrol', 'LPG', 'CNG'], range(4)))
    return repl[fuel]
    
data['fuel'] = data['fuel'].apply(map_fuel) #заменила тип топлива на цифры

#продавец, наверное, тоже важен, но под ? (мало ли, какую цену выставит обычный человек и чем он руководствуется, и какую комиссию накрутят себе салоны)
#пока оставлю, потом подумаю
#print(data['seller_type'].unique()) #'Individual' 'Dealer' 'Trustmark Dealer'

def map_seller_type(seller_type):
    repl = dict(zip(['Individual', 'Dealer', 'Trustmark Dealer'], range(3)))
    return repl[seller_type]
    
data['seller_type'] = data['seller_type'].apply(map_seller_type)

#transmission тоже может влиять на цену

#print(data['transmission'].unique()) #всего два -- 'Manual' 'Automatic' -- тоже можно быстро заменить на цифры
data['transmission'] = data['transmission'].apply(lambda x: 0 if x == 'Manual' else 1)

#с владельцами тоже самое
#print(data['owner'].unique()) #'First Owner' 'Second Owner' 'Third Owner' 'Fourth & Above Owner' 'Test Drive Car'

def map_owner(owner):
    repl = dict(zip(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], range(5)))
    return repl[owner]
    
data['owner'] = data['owner'].apply(map_owner)

#если честно, я не очень поняла, что такое mileage, яндекс переводит как пробег, но у нас уже есть пробег как km_driven, а значения там какие-то странные
data.drop(columns = 'mileage', axis = 1 , inplace = True)

#кажется, что мощность(?) мотора важная вещь, но с этими значениями можно переучиться (будет подгонять), но пока я попробую оставить
#print(data['engine'].unique()) #они все разные, и везде есть "СС" после числа, значит, можно оставить просто число

data['engine'] = data['engine'].str.replace(' CC', '')
data.fillna('0', inplace = True )
data['engine'] = data['engine'].astype('int')

#попробую также оставить макс мощность с надеждой, что оно не переучится
#print(data['max_power'].unique()) #в конце везде стоят bhp

data['max_power'] = data['max_power'].str.replace(' bhp', '')
data['max_power'].astype(bool)
data = data[data['max_power'].astype(bool)]
data['max_power'] = data['max_power'].astype('float')

#я не поняла, что такое torque и чем оно отличается от мощности, но значения там неприятные 
#print(data['torque'].unique())
data.drop(columns = 'torque', axis = 1 , inplace = True)

#сидения, как мне кажется, будут мало влиять на цену
data.drop(columns = 'seats', axis = 1 , inplace = True)

#теперь осталось самое основное -- name
#print(data['name'].unique()) #их оказалось очень много, потому что на название влияет индивидуальная комплектация, модель, иногда пишут год, и т.д.
#но полностью от марки машины отказаться нельзя, т. к. марка как продающий бренд влияет на цену
#поэтому я подумала, что можно оставить только саму марку без уточнений модели (регулярками)

pattern = '\w+'

def find_pattern (str, pattern):
    if re.search(pattern, str):
        return re.search(pattern, str).group()

data['name'] = data['name'].apply(lambda x: find_pattern(x, pattern))

#теперь можно закодировать марки машин 
#print(data['name'].unique()) 

def map_name(name):
    repl = dict(zip(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
 'Tata', 'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes', 'Mitsubishi', 'Audi',
 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
 'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Peugeot'], range(32)))
    return repl[name]

data['name'] = data['name'].apply(map_name)

#как сейчас выглядит датасет после всех действий: 
print(data.head())
print(data.size)
print(data.shape)
print(data.dtypes)
print(data.dtypes.value_counts())
print(data.info())

#только почему-то получилось, что колонки с 0 по 7 (до owner) тип int54,
#engine которому я присваивала тип получился int32
#а max_power float64

y = data['selling_price']
X = data.drop(columns = 'selling_price', axis = 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)

#print(X.shape)

model = LinearRegression()

model.fit(Xtrain, ytrain)

pred_test = model.predict(Xtest) #предсказание
print(mean_squared_error(pred_test, ytest) ** 0.5) 

pred_train = model.predict(Xtrain)
print(mean_squared_error(pred_train, ytrain) ** 0.5)

data_1 = pd.DataFrame(data=np.c_[data.drop(columns = 'selling_price', axis = 1), data['selling_price']],
                     columns=list(data.drop(columns = 'selling_price', axis = 1)) + ['selling_price'])

plt.figure(figsize=(15,15))

corr = data_1.corr()

g = sns.heatmap(corr,annot=True,linewidths=.5,fmt= '.2f',\
            mask=np.zeros_like(corr, dtype=bool), \
            cmap=sns.diverging_palette(200,300,as_cmap=True))

print(data_1.head())

g.set_xticklabels(g.get_xticklabels(), fontsize = 7)
g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 7)
print(plt.show())


for c in data_1.columns:
    if c != 'selling_price':
        print(c)
        plt.scatter(data_1[c], data_1['selling_price'],
                    c = 'm',
                    s = 3)
        plt.show()