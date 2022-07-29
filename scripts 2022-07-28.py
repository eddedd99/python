#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      EDDLAP
#
# Created:     18/11/2021
# Copyright:   (c) EDDLAP 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#Para revisar la estructura de un directorio
tree C:\Users\EDDLAP\Documents\Python\django3\ /F

#Revisar la versión de python instalada
python --version

#ngresar a Pyhton
C:\>python

#Instalar Django
pip install Django==3.2

#Para revisar la versión instalada de Django (3 opciones)
python -m django --version
django-admin --version
python -c "import django; print(django.get_version())"

#Para checar la variable PATH del sistema
echo %PATH:;=&echo.%

#Paso 1. Crear un nuevo proyecto
C:\>django-admin startproject myproject

#Paso 1b. Visualizar lass funciones disponibles del manage.py
C:\>\en la carpeta donde instalo myprojecy\python manage.py help

#Paso 2. Ejecutar el servidor
python manage.py runserver

#Paso 3. Crear la aplicación
python manage.py startapp myapp

#Paso 4. En el archivo settings.py ubicado en carpeta 'myproject'
#actualizar en la sección INSTALLED_APPS el siguiente archivo:

##INSTALLED_APPS = [
##    'django.contrib.admin',
##    'django.contrib.auth',
##    'django.contrib.contenttypes',
##    'django.contrib.sessions',
##    'django.contrib.messages',
##    'django.contrib.staticfiles',
##	'myapp',    <<<<-----ESTA LINEA SE DEBE AGREGAR
##]

#Paso 5. Verificar que se encuentren así los siguientes apartados

##INSTALLED_APPS = (
##   'django.contrib.admin',
##   'django.contrib.auth',
##   'django.contrib.contenttypes',
##   'django.contrib.sessions',
##   'django.contrib.messages',
##   'django.contrib.staticfiles',
##   'myapp',
##)
##
##MIDDLEWARE_CLASSES = (
##   'django.contrib.sessions.middleware.SessionMiddleware',
##   'django.middleware.common.CommonMiddleware',
##   'django.middleware.csrf.CsrfViewMiddleware',
##   'django.contrib.auth.middleware.AuthenticationMiddleware',
##   'django.contrib.messages.middleware.MessageMiddleware',
##   'django.middleware.clickjacking.XFrameOptionsMiddleware',
##)
##

#Paso 6. Cambiarse a la siguiente ruta y ejecutar
C:\Users\EDDLAP\Documents\Python\django\myproject>python manage.py migrate

#Paso 7. Crear un super usuario
##C:\Users\EDDLAP\Documents\Python\django\myproject>python manage.py createsuperuser
##Username (leave blank to use 'eddlap'):
##Email address: eddlap@mail.com
##Password: edd
##Password (again): edd
##This password is too short. It must contain at least 8 characters.
##Bypass password validation and create user anyway? [y/N]: y
##Superuser created successfully.

#Paso 7. En la ruta myproject/url.py verificaar que se tenga (parecido)

##from django.conf.urls import patterns, include, url
##
##from django.contrib import admin
##admin.autodiscover()
##
##urlpatterns = patterns('',
##   # Examples:
##   # url(r'^$', 'myproject.views.home', name = 'home'),
##   # url(r'^blog/', include('blog.urls')),
##
##   url(r'^admin/', include(admin.site.urls)),
##)

#def find_max(nums):
#max_num = float("-inf") # smaller than all other numbers
#for num in nums:
#if num > max_num:
#return max_num

#imprime la ruta directorio actual
import sys
print(sys.path)

#-----------------------------------------------
#2021-11-30 Importar archivo CSV mediante pandas
import pandas as pd

#load dataframe from csv
df = pd.read_csv("C:\\Users\\EDDLAP\\Documents\\Python\\archivo_plano\\archivo_plano2.csv", nrows=20) #lee primeras 20 lineas

#print dataframe
print(df)

#print dataframe 5 rows
print(df.head(5))

#print dataframe tail 5 rows
print(df.tail(5))

#count rows in dataframe
print (df.shape[0])

# Get total all values in column 'importe' of the DataFrame
total = df['Importe'].sum()
print(total)

#filtrar columnas
out = df['Importe'].isin(range(2000,3000))

#filter dataframe
#https://www.listendata.com/2019/07/how-to-filter-pandas-dataframe.html
filtered_df = df[out]
print(filtered_df)

#filtrar columnas tipo==A o tipo==C
newdf = df[df.tipo.isin(["A", "C"])]

print(newdf)

#filtrar diferente A y importe >= 120
newdf = df.loc[(df.tipo != "A") & (df.importe >= 120)]

print(newdf)

#https://www.listendata.com/2019/07/how-to-filter-pandas-dataframe.html
#filtrar opuesto: diferente A y activo >= 1
newdf = df.loc[~((df.tipo == "A") & (df.activo >= 1))]

print(newdf)

print(df)

#interar sobre el dataframe (segundo renglon)
row = df.iloc[1] #index=1 => second row
length = row.size
for i in range(length):
    print(row[i])


#https://www.listendata.com/2019/07/how-to-filter-pandas-dataframe.html
#imprime quienes tienen A o B en el nombre
print(df[df['nombre'].str.contains('A|B')])

#imprime quienes tienen RA o BV en el nombre
print(df[df['nombre'].str.contains('RA|BV')])

#----------------------------------------------------------
#2021-Dic-03
# Importar datos a un dataframe usando pandas
import pandas as pd

# Create a sample pandas DataFrame object
df = pd.DataFrame({'ColA': [1, 1, 1, 1, 1, 2, 3],
                   'ColB': [1, 1, 1, 2, 2, 1, 2],
                   'ColC': ['A','A','A','B','B','C','C']})

# Print the created pandas DataFrame
print('Ejemplo de DataFrame:\n')
print(df)

#Contar filas únicas 2 columnas
print (df.groupby(['ColA','ColB']).size().reset_index(name='Count'))

#Contar filas únicas 3 columnas
print (df.groupby(['ColA','ColB','ColC']).size().reset_index(name='Count'))

#exportar a csv
df.to_csv (r'C:\\Users\\EDDLAP\\Documents\\Python\\archivo_plano\\export.csv', index = False, header=True)

print (df)

#Contar los valores distintos de una Columna
print(df.ColC.value_counts())

#Para exportar a Excel
#instalar pip install openpyxl para que funcione

#Dataframe seleccionar solo 2 columnas
df1 = pd.DataFrame(df, columns = ['ColA', 'ColC'])

print(df1)

#exportar a Excel
df1.to_excel (r'C:\\Users\\EDDLAP\\Documents\\Python\\archivo_plano\\export_excel.xlsx', index = False, header=True)

print (df1)

#----------------------------------------------------------
#https://www.pythonprogramming.in/how-to-get-a-list-of-the-column-headers-from-a-pandas-dataframe.html
#Renombrar las columnas de un Dataframe
import pandas as pd

employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp00'],
    'Name': ['John Doe', 'William Spark'],
    'Occupation': ['Chemist', 'Statistician'],
    'Date Of Join': ['2018-01-25', '2018-01-26'],
    'Age': [23, 24]})

print(employees)

employees.columns = ['EmpCode', 'EmpName', 'EmpOccupation', 'EmpDOJ', 'EmpAge']

print(employees)

#Impresión de las columnas de un Dataframe
print(list(employees))
print(list(employees.columns.values))
print(employees.columns.tolist())
#----------------------------------------------------------

#Seleccionar ciertas filas filtrando los registros
import pandas as pd

df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])

print("\n -- Selecting a single row with .loc with a string -- \n")
print(df.loc['Penelope'])

print("\n -- Selecting multiple rows with .loc with a list of strings -- \n")
print(df.loc[['Cornelia', 'Jane', 'Dean']])

print("\n -- Selecting multiple rows with .loc with slice notation -- \n")
print(df.loc['Aaron':'Dean'])

#-------------------------------------------------------------------------
#https://www.pythonprogramming.in/how-to-add-an-extra-row-at-end-in-a-pandas-dataframe.html
#Agregar una linea al final del Dataframe

import pandas as pd

employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})

print("\n------------ BEFORE ----------------\n")
print(employees)

#Agregar una linea al final
employees.loc[len(employees)] = ['Emp0045', 'Sunny', 'Programmer', '2018-01-25', '45']

print("\n------------ AFTER ----------------\n")
print(employees)

#------------------------------------------------------------------------------------

#Ordenar registros
import pandas as pd

employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})

print(employees)

print(employees.sort_index(axis=1, ascending=False))


#---------------------------------------------------------------------------
#https://www.pythonprogramming.in/how-to-check-the-data-type-of-dataframe-columns-in-pandas.html

import pandas as pd

df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])

print(df.dtypes)

#-------------------------------------------------------------------------------
#Remove duplicate rows

import pandas as pd

df = pd.DataFrame({'Age': [30, 30, 22, 40, 20, 30, 20, 25],
                   'Height': [165, 165, 120, 80, 162, 72, 124, 81],
                   'Score': [4.6, 4.6, 9.0, 3.3, 4, 8, 9, 3],
                   'State': ['NY', 'NY', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])

print("\n -------- Duplicate Rows ----------- \n")
print(df)

df1 = df.reset_index().drop_duplicates(subset='index',
                                       keep='first').set_index('index')

print("\n ------- Unique Rows ------------ \n")
print(df1)

#----------------------------------------------------------------------
#Funciones de fecha hora
import datetime as pd
print(pd.datetime.now())
print(pd.datetime.now().date())
print(pd.datetime.now().year)
print(pd.datetime.now().month)
print(pd.datetime.now().day)
print(pd.datetime.now().hour)
print(pd.datetime.now().minute)
print(pd.datetime.now().second)
print(pd.datetime.now().microsecond)

#---------------------------------------------------------------------
#Leer columnas específicas de un CSV
import pandas as pd

df = pd.read_csv("C:\\Users\\EDDLAP\\Documents\\Python\\test.csv", usecols = ['Wheat','Corn'])
print(df)

#----------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set the interactive mode to ON
plt.ion()

# Check the current status of interactive mode
print(mpl.is_interactive())

#-----------------------------------------------------------------------
import matplotlib.pyplot as plt

#Plot a line graph
plt.plot([5, 15])

# Add labels and title
plt.title("Interactive Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

#Lector de PDF
#------------------------------------------------
import PyPDF2
from PyPDF2 import PdfFileReader

# Creating a pdf file object.
pdf = open("C:\\Users\\EDDLAP\\Documents\\Python\\test.pdf", "rb")

# Creating pdf reader object.
pdf_reader = PyPDF2.PdfFileReader(pdf)

# Checking total number of pages in a pdf file.
print("Total number of Pages:", pdf_reader.numPages)

# Creating a page object. 0 = pagina uno
page = pdf_reader.getPage(0)

# Extract data from a specific page number.
print(page.extractText())

dataf = page.extractText()

print(dataf)

# Closing the object.
pdf.close()

#------------------------------------------------
#Lector de archivo texto plano (contar palabras)

##import pandas as pd
##from sklearn.feature_extraction.text import CountVectorizer
##
### Sample data for analysis
##data1 = "Machine language is a low-level programming language. It is easily understood by computers but difficult to read by people. This is why people use higher level programming languages. Programs written in high-level languages are also either compiled and/or interpreted into machine language so that computers can execute them."
##data2 = "Assembly language is a representation of machine language. In other words, each assembly language instruction translates to a machine language instruction. Though assembly language statements are readable, the statements are still low-level. A disadvantage of assembly language is that it is not portable, because each platform comes with a particular Assembly Language"
##
##df1 = pd.DataFrame({'Machine': [data1], 'Assembly': [data2]})
##
### Initialize
##vectorizer = CountVectorizer(ngram_range=(2, 2))
##doc_vec = vectorizer.fit_transform(df1.iloc[0])
##
### Create dataFrame
##df2 = pd.DataFrame(doc_vec.toarray().transpose(),
##                   index=vectorizer.get_feature_names())
##
### Change column headers
##df2.columns = df1.columns
##print(df2)

#---------------------------------------------------------
import requests

# Download the book
response = requests.get('http://www.gutenberg.org/cache/epub/42671/pg42671.txt')
text = response.text

# Look at some text in the middle
#print(text[4100:4600])

vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform([text])
counts = pd.DataFrame(matrix.toarray(),
                      columns=vectorizer.get_feature_names())

# Show us the top 10 most common words
counts.T.sort_values(by=0, ascending=False).head(10)

#-----------------------------------------------------------------
#Tokenizar un texto
# Importing necessary library
import pandas as pd
import numpy as np
import nltk
import os
import nltk.corpus
import nltk
nltk.download('punkt')

# sample text for performing tokenization
text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of South America"
# importing word_tokenize from nltk
from nltk.tokenize import word_tokenize
# Passing the string text into word tokenize for breaking the sentences
token = word_tokenize(text)
print(token)

#conteo
from nltk.probability import FreqDist
fdist = FreqDist(token)
print(fdist)

# To find the frequency of top 10 words
fdist1 = fdist.most_common(10)
print(fdist1)

#-------------------------------------------------------
#https://oxylabs.io/blog/python-web-scraping
#Scrap sitio web
import requests
response = requests.get("https://oxylabs.io/")
print(response.text)

#ejemplo2 Imprime el elemento título
import requests
url='https://oxylabs.io/blog'
response = requests.get(url)
print(response)

from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.title)

#------------------------------------------------------------------------------------------
#https://matplotlib.org/stable/plot_types/basic/plot.html#sphx-glr-plot-types-basic-plot-py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
x = np.linspace(0, 10, 100)
y = 4 + 2 * np.sin(2 * x)

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()

#-------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()

#-------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')

# make data:
np.random.seed(1)
x = np.random.uniform(-3, 3, 256)
y = np.random.uniform(-3, 3, 256)
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
levels = np.linspace(z.min(), z.max(), 7)

# plot:
fig, ax = plt.subplots()

ax.plot(x, y, 'o', markersize=2, color='lightgrey')
ax.tricontour(x, y, z, levels=levels)

ax.set(xlim=(-3, 3), ylim=(-3, 3))

plt.show()

#-----------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np


code = np.array([
    1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,
    0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1])

pixel_per_bar = 4
dpi = 100

fig = plt.figure(figsize=(len(code) * pixel_per_bar / dpi, 2), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
ax.set_axis_off()
ax.imshow(code.reshape(1, -1), cmap='binary', aspect='auto',
          interpolation='nearest')
plt.show()

#-----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()

#-------------------------------------------------------
import pandas as pd

# The URL we will read our data from
url = 'https://en.wikipedia.org/wiki/List_of_countries_by_meat_consumption'
# read_html returns a list of tables from the URL
tables = pd.read_html(url)

# The data is in the first table - this changes from time to time - wikipedia is updated all the time.
table = tables[0]

print(table.head())

#-------------------------------------------------------
import pandas as pd
import geopandas

# The URL we will read our data from
url = 'https://en.wikipedia.org/wiki/List_of_countries_by_meat_consumption'
# read_html returns a list of tables from the URL
tables = pd.read_html(url)

# The data is in the first table - this changes from time to time - wikipedia is updated all the time.
table = tables[0]

# Read the geopandas dataset
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# Merge the two DataFrames together
table = world.merge(table, how="left", left_on=['name'], right_on=['Country'])

print(table.head())

#-------------------------------------------------------
#2022-Mar-23

# importing required modules
import PyPDF2

#Now give the pdf name
pdfFileObj = open('C:\\Users\\EDDLAP\\Documents\\Python\\gst-revenue-collection-march2020.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

#dame el numero de páginas
print(pdfReader.numPages) # will give total number of pages in pdf

#------------------------------------------------------
#2022-Mar-23

# import packages
import PyPDF2
import re

# open the pdf file
object = PyPDF2.PdfFileReader("C:\\Users\\EDDLAP\\Documents\\Python\\prueba.pdf")

# get number of pages
NumPages = object.getNumPages()

# define keyterms
String = "ínea"

# extract text and do the search
for i in range(0, NumPages):
    PageObj = object.getPage(i)
    print("this is page " + str(i))
    Text = PageObj.extractText()
    #print(Text)
    ResSearch = re.search(String, Text)
    print(ResSearch)

#--------------------------------------------------------------------------------------------
#23-Mar-22
#Buscar un texto en un archivo PDF

# import packages
import PyPDF2
import re

# If you get the PdfReadError: Multiple definitions in dictionary at byte, add strict = False
pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False)

# if you get the UnicodeEncodeError: 'charmap' codec can't encode characters, add .encode("utf-8") to your text
text = pageObj.extractText().encode('utf-8')
pdfFileObj = open('C:\\Users\\EDDLAP\\Documents\\Python\\prueba.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False)

search_word = "rueba"
search_word_count = 0

for pageNum in range(0, pdfReader.numPages):
    pageObj = pdfReader.getPage(pageNum)
    text = pageObj.extractText().encode('utf-8')
    search_text = text.lower().split()
    for word in search_text:
        if search_word in word.decode("utf-8"):
            search_word_count += 1

print("The word {} was found {} times".format(search_word, search_word_count))

#-------------------------------------------------------------------
#https://www.codingem.com/python-find-a-word-in-a-file/
#Buscar una cadena en un archivo de texto

import time
from datetime import timedelta
start = time.time()

#word = "AESP781002MMCNNT07"
word = "CULE780128HDFRPD06"
with open("C:\\cygwin64\\home\\EDDLAP\\imss\\imss_trab_202109.txt", "r") as file:

#word = "palacio"
#with open("C:\\Users\\EDDLAP\\Documents\\Python\\ejemplo.txt", "r") as file:
    for line_number, line in enumerate(file, start=1):
        if word in line:
          print(f"Word '{word}' found on line {line_number}")
          break
end = time.time()
elapsed_time = end-start

print("Busqueda completada: " + str(timedelta(seconds=elapsed_time)))

#----------------------------------------------------------------------
# 13/05/2022 12:30:05 p. m.
# -------------------------
# creating a variable and storing the text
# that we want to search
search_text = "dummy"

# creating a variable and storing the text
# that we want to add
replace_text = "replaced"

# Opening our text file in read only
# mode using the open() function
with open(r'C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\SampleFile.txt', 'r') as file:

    # Reading the content of the file
    # using the read() function and storing
    # them in a new variable
    data = file.read()

    # Searching and replacing the text
    # using the replace() function
    data = data.replace(search_text, replace_text)

# Opening our text file in write only
# mode to write the replaced content
with open(r'C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\SampleFile2.txt', 'w') as file:

    # Writing the replaced data in our
    # text file
    file.write(data)

# Printing Text replaced
print("Text replaced")

#--------------------------------------------------
# 13/05/2022 12:44:56 p. m.
# Reemplazar texto utilizando expresiones regulares
# -------------------------------------------------
# Importing re module
import re

# Creating a function to
# replace the text
def replacetext(search_text,replace_text):

	# Opening the file in read and write mode
	with open('C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\SampleFile.txt','r+') as f:

		# Reading the file data and store
		# it in a file variable
		file = f.read()

		# Replacing the pattern with the string
		# in the file data
		file = re.sub(search_text, replace_text, file)

		# Setting the position to the top
		# of the page to insert data
		f.seek(0)

		# Writing replaced data in the file
		f.write(file)

		# Truncating the file size
		f.truncate()

	# Return "Text replaced" string
	return "Text replaced"

# Creating a variable and storing
# the text that we want to search
search_text = "dummy"

#Creating a variable and storing
# the text that we want to update
replace_text = "replaced"

# Calling the replacetext function
# and printing the returned statement
print(replacetext(search_text,replace_text))


#--------------------------------
# 13/05/2022 13:01:11 p. m.
# Reemplazar texto en archivo csv
# -------------------------------
# creating a variable and storing the text
# that we want to search
search_text = "Ǹ"

# creating a variable and storing the text
# that we want to add
replace_text = "é"

# Opening our text file in read only
# mode using the open() function
# with open('C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\1 fuentes\\test.csv', 'r') as file:

with open('C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\1 fuentes\\afluenciastc_desglosado_03_2022_V2.csv', 'r') as file:


    # Reading the content of the file
    # using the read() function and storing
    # them in a new variable
    data = file.read()

    # Searching and replacing the text
    # using the replace() function
    data = data.replace(search_text, replace_text)

# Opening our text file in write only
# mode to write the replaced content
with open('C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\1 fuentes\\test2.csv', 'w') as file:

    # Writing the replaced data in our
    # text file
    file.write(data)

# Printing Text replaced
print("Text replaced")



#--------------------------------------------------
# 13/05/2022 13:05:00 p. m.
# Reemplazar texto utilizando expresiones regulares
# -------------------------------------------------
# Importing re module
import re

# Creating a function to
# replace the text
def replacetext(search_text,replace_text):

	# Opening the file in read and write mode
	with open('C:\\Users\\EDDLAP\\Documents\\Proyectos\\STCMetro\\1 fuentes\\afluenciastc_desglosado_03_2022_V2.csv','r+',encoding = 'latin-1') as f:

		# Reading the file data and store
		# it in a file variable
		file = f.read()

		# Replacing the pattern with the string
		# in the file data
		file = re.sub(search_text, replace_text, file)

		# Setting the position to the top
		# of the page to insert data
		f.seek(0)

		# Writing replaced data in the file
		f.write(file)

		# Truncating the file size
		f.truncate()

	# Return "Text replaced" string
	return "Text replaced"

# Creating a variable and storing
# the text that we want to search
search_text = "Ǹ"

#Creating a variable and storing
# the text that we want to update
replace_text = "é"

# Calling the replacetext function
# and printing the returned statement
print(replacetext(search_text,replace_text))


#--------------------------------------------------
# 24/05/2022 10:22:00 p. m.
# PDF leer archivo
# -------------------------------------------------
#
import PyPDF2
from PyPDF2 import PdfFileReader

# Creating a pdf file object.
pdf = open("C:\\Users\\EDDLAP\\Documents\\Proyectos\\BoletinJud\\Laboral\\b02ene19.pdf", "rb")

# Creating pdf reader object.
pdf_reader = PyPDF2.PdfFileReader(pdf)

# Checking total number of pages in a pdf file.
print("Total number of Pages:", pdf_reader.numPages)

# Creating a page object. 0 = pagina uno
page = pdf_reader.getPage(3)

# Extract data from a specific page number.
print(page.extractText())

dataf = page.extractText()

print(dataf)

# Closing the object.
pdf.close()

# Extraer texto de PDF con alta precisión (tokenizado)
# https://www.analyticsvidhya.com/blog/2021/06/data-extraction-from-unstructured-pdfs/
# https://pypi.org/project/PyMuPDF/
#
# 1. Instalar primero modulo PyMuPDF
# python -m pip install --upgrade pymupdf

import fitz
import pandas as pd
doc = fitz.open('C:\\Users\\EDDLAP\\Documents\\Python\\leer_pdf\\personas1.pdf')
page1 = doc[0]
words = page1.get_text("words")
print(words)

#-------------------------------------------
#2022-07-13
#Pasar argumentos en Python linea de comando
import sys

n = len(sys.argv)
print("Total arguments passed:", n)

print("\nName of Python script:", sys.argv[0])

print("\nArguments passed:", end = " ")
for i in range(1, n):
    print(sys.argv[i], end = " ")

Sum = 0
for i in range(1, n):
    Sum += int(sys.argv[i])

print("\n\nResult:", Sum)

#---------------------------------------------
#2022-07-13
#Leer metadata de un archivo MP3


from mutagen.mp3 import MP3

audio = MP3("C:\\Users\\EDDLAP\\Documents\\Python\\metadata_audio\\ejemplo.mp3")
print("info length: " + str(audio.info.length))
print("   bit rate: " + str(audio.info.bitrate))
print(audio.keys)


#--------------------------------------------
#Verificar version de python
python --version

#Verificar version de pip
pip -V

#Actualizar PIP
python -m pip install --upgrade pip

#Instalar Tkinter
pip install tk

#---------------------------------------------
#Webscraping
#https://www.dataquest.io/blog/web-scraping-python-using-beautiful-soup/
#Verificar el estatus página 200=OK

import requests
#page = requests.get("https://dataquestio.github.io/web-scraping-pages/simple.html")
page = requests.get("https://realmenuprices.com/famous-daves-menu-prices/")
print(page.status_code)
print(page.content)

from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())

list(soup.children)


#--------------------------------------------------
#2022-jul-13
#Webscraping
#https://jarroba.com/scraping-python-beautifulsoup-ejemplos/

# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

from bs4 import BeautifulSoup
import requests

URL_BASE = "http://jarroba.com/"
MAX_PAGES = 20
counter = 0

for i in range(1, MAX_PAGES):

    # Construyo la URL
    if i > 1:
        url = "%spage/%d/" % (URL_BASE, i)
    else:
        url = URL_BASE

    # Realizamos la petición a la web
    req = requests.get(url)
    # Comprobamos que la petición nos devuelve un Status Code = 200
    statusCode = req.status_code
    if statusCode == 200:

        # Pasamos el contenido HTML de la web a un objeto BeautifulSoup()
        html = BeautifulSoup(req.text, "html.parser")

        # Obtenemos todos los divs donde estan las entradas
        entradas = html.find_all('div', {'class': 'col-md-4 col-xs-12'})

        # Recorremos todas las entradas para extraer el título, autor y fecha
        for entrada in entradas:
            counter += 1
            titulo = entrada.find('span', {'class': 'tituloPost'}).getText()
            autor = entrada.find('span', {'class': 'autor'}).getText()
            fecha = entrada.find('span', {'class': 'fecha'}).getText()

            # Imprimo el Título, Autor y Fecha de las entradas
            # print "%d - %s  |  %s  |  %s" % (counter, titulo, autor, fecha)
            # print (counter, titulo, autor, fecha)
            resultado = str(counter) + '||' + titulo + '||' + autor
            print (resultado)

    else:
        # Si ya no existe la página, me da un 400
        break


#------------------------------------------------
#webscraping
#2022-jul-13

#import requests
#--page = requests.get("https://dataquestio.github.io/web-scraping-pages/simple.html")
#page = requests.get("C:\\Users\\EDDLAP\\Documents\\Python\\webscraping2\example.html")
#print(page.status_code)
#print(page.content)

from bs4 import BeautifulSoup
f = open("C:\\Users\\EDDLAP\\Documents\\Python\\webscraping2\example.html", encoding="utf8")
soup = BeautifulSoup(f)
print(soup.prettify())
f.close()

----------------------------------------------------------------------
#2022-jul-14
Tutorial de numpy
https://www.machinelearningplus.com/python/101-numpy-exercises-python/


----------------------------------------------------------------------
#2022-jul-18

value = input("Please enter dollars:\n")
value = int(value)
# print(f'You entered {value}')

if (value >= 100):
    print ("YES")
else:
    print ("NO")


----------------------------------------------------------------------
#2022-jul-18

i = 1
while i < 6:
  print(i)
  i += 1
  print("h")
else:
  print("i is no longer less than 6")

----------------------------------------------------------------------
#El cuerpo del bucle en Python termina siempre antes del incrementador
numero=1
suma=1
while numero <= 10:
    suma = numero + suma
    print ("La suma es " + str(suma))
    numero = numero + 1 #aqui termina bucle
----------------------------------------------------------------------
#Leer input separados por espacio en blanco

print("Enter the numbers: ")
inp = list(map(int, input().split()))
print(inp)

----------------------------------------------------------------------
#2022-07-18
#Problema del Chef y comensal (codechef.com)

#Leer num veces
input_veces = input("veces?: ")
input_veces = int(input_veces)

#Para cada una de las veces
i = 1
while i <= input_veces:
      print("Ingresa par de numeros:")
      numeros = list(map(int, input().split()))
      X = int(numeros[0])
      Y = int(numeros[1])

      if (X >= Y):
          print ("YES")
      else:
          print ("NO")

      i = i + 1

-----------------------------------------------------------
#Encontrar los dígitos

new_string = "Germany26China47Australia88"

emp_str = ""
for m in new_string:
    if m.isdigit():
        emp_str = emp_str + m
print("Find numbers from string:",emp_str)

-----------------------------------------------------------
#Sumar los digitos en una cadena

new_string = "12345"

total_dig=0
emp_str = ""
for m in new_string:
    if m.isdigit():
        total_dig=total_dig + int(m)
print("Suma de digitos:",total_dig) # suma 1 2 3 4 5 => 15

-----------------------------------------------------------
#Agregar a una lista

new_str = "Micheal 89 George 94"

emp_lis = []
for z in new_str.split():
   if z.isdigit():
      emp_lis.append(int(z))

print("Find number in string:",emp_lis)

------------------------------------------------------------
#Regular expression

import re

new_string = 'Rose67lilly78Jasmine228Tulip'
new_result = re.findall('[0-9]+', new_string)
print(new_result)

---------------------------------------------------------------------------
#2022-07-27
#Buscar una cadena de texto en un archivo
string1 = 'prueba'

# opening a text file
file1 = open("C:\\Users\\EDDLAP\\Documents\\Python\\texto\\texto.txt", "r")

# setting flag and index to 0
flag = 0
index = 0

# Loop through the file line by line
for line in file1:
    index += 1

    # checking string is present in line or not
    if string1 in line:

      flag = 1
      break

# checking condition for string found or not
if flag == 0:
   print('String', string1 , 'Not Found')
else:
   print('String', string1, 'Found In Line', index)

# closing text file
file1.close()

---------------------------------------------------------------------------
#2022-07-27
#Buscar una cadena de texto en un archivo del IMSS (FUNCIONA OK)

import datetime as pd
ini = pd.datetime.now()
print(ini)

#String to search
string1 = 'CULE780128HDFRPD06'

# opening a text file
file1 = open("C:\\cygwin64\\home\\EDDLAP\\imss\\imss_trab_202109.txt", "r", encoding="ANSI")

# setting flag and index to 0
flag = 0
index = 0

# Loop through the file line by line
for line in file1:
    index += 1

    # checking string is present in line or not
    if string1 in line:

      flag = 1
      break

# checking condition for string found or not
if flag == 0:
   print('String', string1 , 'Not Found')
else:
   print('String', string1, 'Found In Line', index)

# closing text file
file1.close()

#Fin
fin = pd.datetime.now()
print(fin)
print(fin - ini)

--------------------------------------------------------------------
#2022-07-28
#Query a un archivo CSV grande con formato ANSI y usando módulo CSV

import csv

filename = "C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\SQL\\compras.csv"
#filename = "C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\DAX\\data.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open(filename, 'r',  encoding="ANSI") as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))

--------------------------------------------------------------------------------
#2022-07-28
#Contar valores únicos en archivo CSV Agrupados por InvoiceNo

import pandas as pd

# define path to data
PATH = u'C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\SQL\\compras.csv'

# create panda datafrmae
df = pd.read_csv(PATH, usecols = [0,1,2,3,4,5,6,7], header = 1, names = ['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country'])

#Visualizar dataframe
print(df)

#Cuantos registros por Country
print(df.Country.value_counts())

#Cuantas facturas diferentes existen
print(df.InvoiceNo.value_counts())

# Add count to column of interest
df['count'] = df.groupby('Country')['InvoiceNo'].transform('count')

#Visualizar dataframe (sin truncado)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df)

#Visualizar dataframe (reseteando de nuevo)
pd.reset_option('all')
print(df)

#Salida del print a un archivo
f = open('C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\SQL\\salidaNX.csv','w')

#Contar valor unicos por linea, guardar en archivo (https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file)
print(df.nunique(axis=1), file=f)
f.close()

#Exportar a csv
#https://towardsdatascience.com/how-to-export-pandas-dataframe-to-csv-2038e43d9c03

#index=true numera la 1a columna
df.to_csv('C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\SQL\\salida.csv', index=False)

#Exportar a csv con tabulador
df.to_csv('C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\SQL\\salida2.csv', index=True,sep='\t')

#Exportar a csv con tabulador
df.to_csv('C:\\Users\\EDDLAP\\Documents\\Proyectos\\Power BI\\Ejemplos\\SQL\\salida2.csv', index=True,sep=';')

--------------------------------------------------------------------------------------------------------------
#2022-07-28
#https://pandas.pydata.org/pandas-docs/dev/user_guide/10min.html
#Uso de Pandas

import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])

print(s)

dates = pd.date_range("20130101", periods=6)

print(dates)

#Crear dataframe
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

#Imprime
df

#Da indice
df.index

#Da columnas
df.columns

#Proporciona min, max, stdv
df.describe()

#Transponer
df.T

#Ordenar por indice
df.sort_index(axis=1, ascending=False)

#Ordenar por columna
df.sort_values(by="B")

#Seleccionar col A
df.A

#Seleccionar col A
df["A"]

#Seleccionar 3 registros
df[0:3]

#Seleccionar cols A y B
df.loc[:, ["A", "B"]]

#Seleccionar ciertas fechas, cols A y B
df.loc["20130102":"20130104", ["A", "B"]]

#Seleccionar valor específico
df.iloc[1, 1]

df

#Seleccionar col A > 0
df[df["A"] > 0]

#Seleccionar df > 0
df[df > 0]

#Calcular promedio
df.mean()

#-----------------------------------------------------------------------------------
#2022-07-28
#Grafico series de tiempo

import matplotlib.pyplot as plt
plt.close("all")

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot();
plt.show();


----------------------------------------------------------------------------------------------------------
#2022-07-28
#Importar multiples CSVs en un solo dataset
import glob
import pandas as pd

df = pd.concat(map(pd.read_csv, glob.glob('C:\\Users\\EDDLAP\\Documents\\Proyectos\\Python\\csv\\*.csv')))

#df = pd.concat(map(pd.read_csv, ['C:\\Users\\EDDLAP\\Documents\\Proyectos\\Python\\csv\\data1.csv', 'C:\\Users\\EDDLAP\\Documents\\Proyectos\\Python\\csv\\data2.csv','C:\\Users\\EDDLAP\\Documents\\Proyectos\\Python\\csv\\data3.csv']))

#Imprime columnas
print(df)

#Reindexar
df = df.reset_index()

#imprime columnas
df.columns

#borrar columna
del df["index"]

-------------------------------------------------------------------------------------
#2022-07-28
#Eliminar filas de un dataframe
#https://www.freecodecamp.org/news/drop-list-of-rows-from-pandas-dataframe/

import pandas as pd

data = {"product_name":["Keyboard","Mouse", "Monitor", "CPU","CPU", "Speakers",pd.NaT],
        "Unit_Price":[500,200, 5000.235, 10000.550, 10000.550, 250.50,None],
        "No_Of_Units":[5,5, 10, 20, 20, 8,pd.NaT],
        "Available_Quantity":[5,6,10,"Not Available","Not Available", pd.NaT,pd.NaT],
        "Available_Since_Date":['11/5/2021', '4/23/2021', '08/21/2021','09/18/2021','09/18/2021','01/05/2021',pd.NaT]
       }

df = pd.DataFrame(data)

df

#Eliminar registros 5 y 6 del df
#axis=0 significa que se haga sobre rows
#inplace=True significa sobre el mismo dataset

df.drop([5,6], axis=0, inplace=True)

df

----------------------------------------------------------
#Exportar datafram a html
#https://pythonexamples.org/pandas-render-dataframe-as-html-table/#:~:text=To%20render%20a%20Pandas%20DataFrame,thead%3E%20table%20head%20html%20element.
import pandas as pd

#create dataframe
df_marks = pd.DataFrame({'name': ['Somu', 'Kiku', 'Amol', 'Lini'],
     'physics': [68, 74, 77, 78],
     'chemistry': [84, 56, 73, 69],
     'algebra': [78, 88, 82, 87]})

#render dataframe as html
html = df_marks.to_html()

#write html to file
text_file = open("C:\\Users\\EDDLAP\\Documents\\Proyectos\\Python\\csv\\index.html", "w")
text_file.write(html)
text_file.close()

-----------------------------------------------------------------------------------------
#Array
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

----------------------------------
#Normal distribution
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(size=1000), hist=False)

plt.show()

----------------------------------
#Binomial Distribution
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)

plt.show()

#---------------------------------------------------------------------------
#IMSS Consulta

string1 = 'CAMA591009MDFRRN12'

# opening a text file
file1 = open("C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt", "r", encoding="ANSI")

# setting flag and index to 0
flag = 0
index = 0
mysum = 0

# Loop through the file line by line
for line in file1:
    index += 1

    # checking string is present in line or not
    if string1 in line:

      flag = 1
      break

# checking condition for string found or not
if flag == 0:
   print('String', string1 , 'Not Found')
else:
   print('String', string1, 'Found In Line', index)
   print(line)

# closing text file
file1.close()


------------------------------------------------------------------------------------------
#IMSS Consulta (Sumar primera columna)

# opening a text file
file1 = open("C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt", "r", encoding="ANSI")

#obtiene distintos valores 1a. columna
print(set([list(line)[0] for line in file1]))

# setting sum to 0
mysum = 0

#Ignora primera linea
next(file1)

# Loop through the file line by line
for line in file1:
    mysum += int(line.split('|')[0])

print('MySum:', mysum)

# closing text file
file1.close()

------------------------------------------------------------------------------------------
#IMSS Consulta
#https://www.codegrepper.com/code-examples/python/python+read+text+file+line+by+line+into+list

# opening a text file
file1 = open("C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt", "r", encoding="ANSI")

listl=[]
for line in file1:
		strip_lines=line.strip('|')
		listli=strip_lines.split()
		print(listli)
		m=listl.append(listli)
print(listl[2])

---------------------------------------------------------------------------------------------
#2022-07-28
#Leer columna 0 y añadir en lista

file = "C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt"

f=open(file,"r",encoding="ANSI")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split('|')[0])
f.close()
print(result)


---------------------------------------------------------------------------------------------
#2022-07-28
#Leer columna 0 y contar valores distintos (usando Collenctions)

from collections import Counter
file = "C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt"

f=open(file,"r",encoding="ANSI")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split('|')[2])
f.close()
#print(result)

print(Counter(result).keys())
print(Counter(result).values())


---------------------------------------------------------------------------------------------
#2022-07-28
#Leer columna 0 y contar valores distintos (usando Numpy)

import numpy as np
file = "C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt"

f=open(file,"r",encoding="ANSI")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split('|')[0])
f.close()
#print(result)

values, counts = np.unique(result, return_counts=True)
print(values,counts)


---------------------------------------------------------------------------------------------
#2022-07-28
#Leer columna 0 y contar valores distintos (usando Numpy y dataframe)

import numpy as np
import pandas as pd

file = "C:\\cygwin64\\home\\EDDLAP\\imss\\imss_40regV2.txt"
#file = "C:\\cygwin64\\home\\EDDLAP\\imss\\imss_trab_202109.txt"

f=open(file,"r",encoding="ANSI")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split('|')[6])
f.close()
#print(result)

values, counts = np.unique(result, return_counts=True)
#print(values,counts)

#incluir en dataframe
df_results = pd.DataFrame(list(zip(values,counts)),columns=["value","count"])
#print(df_results)

#Contar y ordenar por columna
final_df = df_results.sort_values(by=['count'], ascending=False)
print(final_df)

