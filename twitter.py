import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
import string
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import collections
from scipy.stats.stats import pearsonr
import scipy.stats as stats
import math
import matplotlib.style as style 
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import sklearn.datasets
from PIL import Image
import time
import datetime
import plotly.plotly as py

tweets = pd.read_csv('.../Desktop/Big data/Tweets.csv')

#Gráficas del número de tweets por mes, mes y dia y por horas
c=tweets.groupby(['autor','mes_dia']).sum()
c['numero']=tweets.groupby(['autor','mes_dia']).count()['fecha']
c=c.reset_index()
c['mes_dia'] = pd.to_datetime(c['mes_dia'])
colors=['orange','green','purple','blue','red']

#Se añaden además una serie de eventos clave que permiten explicar el comportamiento
key_events = [(pd.to_datetime('Dec 2 2018'), 'Elecciones Andalucia'),
              (pd.to_datetime('Jun 2 2018'), 'Investidura P. Sanchez'),
              (pd.to_datetime('Jul 21 2018'), 'P. Casado líder PP'),
              (pd.to_datetime('Oct 12 2018'), '12 de Octubre'),
              (pd.to_datetime('Apr 26 2018'), 'Sentencia Manada'),
              (pd.to_datetime('Jan 20 2019'), 'Convencion nacional PP'),
              (pd.to_datetime('Mar 8 2018'), '8 de Marzo'),
              (pd.to_datetime('Oct 29 2017'), 'Huida de puigdemont'),
              (pd.to_datetime('Mar 12 2019'), 'Inicio del Proces')
              ]

#Se grafica el número de mensajes que publica cada autor al día
for i,nombre in zip(range(5),['Albert_Rivera', 'Santi_ABASCAL','Pablo_Iglesias_','pablocasado_','sanchezcastejon']):
    pd.Series(data=c[c['autor']==nombre]['numero'].values,index=c[c['autor']==nombre]['mes_dia']).plot(color=colors[i],x='mes_dia',linewidth=0.5)
    plt.xlabel('Mes_Dia')
    plt.ylabel('Numero de tweets')
    plt.legend(['Albert Rivera', 'Santiago Abascal','Pablo Iglesias','Pablo Casado','Pedro Sanchez'],loc='center left')
for event in key_events:
    plt.axvline(event[0], color='black',linewidth=0.5,linestyle='--')
    plt.text(event[0] + pd.Timedelta(1, 'm'), 65, event[1], rotation=90, size=7)
plt.show()

#Se realiza la misma gráfica mostrando el número de tweets publicados agregados por mes
c=tweets.groupby(['autor','mes']).sum()
c['numero']=tweets.groupby(['autor','mes']).count()['fecha']
c=c.reset_index()
c['mes'] = pd.to_datetime(c['mes'])
for i,nombre in zip(range(5),['Albert_Rivera', 'Santi_ABASCAL','Pablo_Iglesias_','pablocasado_','sanchezcastejon']):
    pd.Series(data=c[c['autor']==nombre]['numero'].values,index=c[c['autor']==nombre]['mes']).plot(color=colors[i],x='mes')
    plt.xlabel('Mes')
    plt.ylabel('Numero de tweets')
    plt.legend(['Albert Rivera', 'Santiago Abascal','Pablo Iglesias','Pablo Casado','Pedro Sanchez'],loc='center left')
for event in key_events:
    plt.axvline(event[0], color='black',linewidth=0.5,linestyle='--')
    plt.text(event[0] + pd.Timedelta(1, 'm'), 850, event[1], rotation=90, size=7)
plt.show()

#Se muestra la presencia de mensajes en diferentes horas del día
c=tweets.groupby(['autor','hora']).sum()
c['numero']=tweets.groupby(['autor','hora']).count()['fecha']
c=c.reset_index()
c['hora'] = pd.to_datetime(c['hora'],format='%H').dt.hour
for i,nombre in zip(range(5),['Albert_Rivera', 'Santi_ABASCAL','Pablo_Iglesias_','pablocasado_','sanchezcastejon']):
    pd.Series(data=c[c['autor']==nombre]['numero'].values,index=c[c['autor']==nombre]['hora']).plot(color=colors[i],x='hora')
    plt.xlabel('hora')
    plt.ylabel('Numero de tweets')
    plt.legend(['Albert Rivera', 'Santiago Abascal','Pablo Iglesias','Pablo Casado','Pedro Sanchez'])
plt.show()

#Proceso de Tokenizacion
RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
def strip_emoji(texto):
    return RE_EMOJI.sub(r'', text)
def tokenizar(texto):
	texto=texto.lower()
	#eliminamos web
	texto=re.sub(r'http\S+',"",texto)
	#elimianos signos puntuacion
	texto=re.sub('[%s]' % re.escape(string.punctuation), ' ', texto)
	#eliminamos numeros
	texto=re.sub(r'\d+','',texto)
	#eliminamos palabras de una letra
	texto=re.sub(r'\b\w\b','',texto)
	#eliminamos otras cosas	
	texto=re.sub('[@#$&¡]','',texto)
	#eliminamos emojis
	texto=RE_EMOJI.sub(r'', texto)
	#eliminamos espacios	
	texto=re.sub(' +',' ',texto)
	return texto

test='   Esto es un ejemplo 34 del PROCESO3 de Tokenizacion ? para @Albert_Rivera & #https://ciudadanos.com #cs'
tokenizado=tokenizar(test)
print('El texto de prueba es: %s' %test + '\nEl texto tokenizado es: %s' %tokenizado)

#Palabras mas comunes
def palabras_comunes(texto):
    contador=collections.Counter()
    for i in range(texto.index[0],texto.index[0]+len(texto.index)-1):
        terminos=[termino for termino in texto[i].split()]
        contador.update(terminos)
    return contador.most_common(30)
print(palabras_comunes(tweets['texto']))

#Numero de tweets de cada uno
print(tweets.groupby('autor').count())

#Añadimos el numero de palabras al conjunto de datos inicial
def numero_palabras(texto):
    palabras=np.arange(len(texto))
    for j in range(len(texto)):
        palabras[j]=len([i for i in texto[j].lower().split()])
    return palabras
tweets['palabras']=numero_palabras(tweets['texto'])
tweets['palabras2']=numero_palabras(tweets['texto_filtrado'])


#Se realiza una comparativa de la importancia de introducir filtrado. Se grafica el número de palabras en un caso y en el otro
plt.fill_between(tweets.groupby(['mes']).sum().reset_index()['mes'],tweets.groupby(['mes']).sum()['palabras'], step="pre", alpha=0.4)
plt.fill_between(tweets.groupby(['mes']).sum().reset_index()['mes'],tweets.groupby(['mes']).sum()['palabras2'], step="pre", alpha=0.4)
tweets.groupby(['mes']).sum()['palabras'].plot(drawstyle="steps",x='mes')
tweets.groupby(['mes']).sum()['palabras2'].plot(drawstyle="steps",x='mes')
plt.xlabel('mes')
plt.ylabel('Numero de palabras')
plt.legend(['Texto con stopwords','Texto sin stopwords'])
plt.show()

#Copmparacion uso de palabras por autor
tweets.boxplot(column='palabras',by='autor')
plt.show()

#Tokenizamos los mensajes y lo introducimos en el conjunto de daos
tweets['texto']=tweets['texto'].map(tokenizar)

#hacemos la nube de palabras, que incluye aquellas mas usadas
Tweet_mask=np.array(Image.open(".../Desktop/Big data/trabajo/twitter_mask.png"))

def wordcloud(tweets,col):
    stopwords1 = set(stopwords.words('spanish'))
    rivera = WordCloud(background_color="white",stopwords=stopwords1,mask = Tweet_mask).generate(" ".join([i for i in tweets[tweets['autor']=='Albert_Rivera'][col]])) #poner mask = Tweet_mask ver https://beginanalyticsblog.wordpress.com/2018/02/07/twitter-data-analysis-using-python/
    iglesias = WordCloud(background_color="white",stopwords=stopwords1).generate(" ".join([i for i in tweets[tweets['autor']=='Pablo_Iglesias_'][col]]))
    casado = WordCloud(background_color="white",stopwords=stopwords1).generate(" ".join([i for i in tweets[tweets['autor']=='pablocasado_'][col]]))
    sanchez = WordCloud(background_color="white",stopwords=stopwords1).generate(" ".join([i for i in tweets[tweets['autor']=='sanchezcastejon'][col]]))
    abascal = WordCloud(background_color="white",stopwords=stopwords1).generate(" ".join([i for i in tweets[tweets['autor']=='Santi_ABASCAL'][col]]))
    plt.figure( figsize=(20,10))
    plt.subplot(2,3,1)
    plt.imshow(rivera)
    plt.axis("off")
    plt.title('Albert Rivera')
    plt.subplot(2,3,2)
    plt.imshow(iglesias)
    plt.axis("off")
    plt.title('Pablo Iglesias')
    plt.subplot(2,3,3)
    plt.imshow(casado)
    plt.title('Pablo Casado')
    plt.axis("off")
    plt.subplot(2,3,4)
    plt.imshow(sanchez)
    plt.title('Pedro Sanchez')
    plt.axis("off")
    plt.subplot(2,3,5)
    plt.imshow(abascal)
    plt.axis("off")
    plt.title('Santiago Abascal')
    plt.show()
wordcloud(tweets,'texto')

#Eliminamos las palabras innecesarias
stop_words=stopwords.words('spanish')
stop2=['va','así','tras','hacer','ser','hecho','acto','hora','apoyo','día','junto','toda','vez','hoy']
for i in range(len(stop2)):
    stop_words.append(stop2[i])
tweets['texto_filtrado']=tweets['texto'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
wordcloud(tweets,'texto_filtrado')

#Palabras más usadas por autor
palabras_rivera=(palabras_comunes(tweets[tweets['autor']=='Albert_Rivera']['texto_filtrado']))
palabras_iglesias=(palabras_comunes(tweets[tweets['autor']=='Pablo_Iglesias_']['texto_filtrado']))
palabras_casado=(palabras_comunes(tweets[tweets['autor']=='pablocasado_']['texto_filtrado']))
palabras_sanchez=(palabras_comunes(tweets[tweets['autor']=='sanchezcastejon']['texto_filtrado']))
palabras_abascal=(palabras_comunes(tweets[tweets['autor']=='Santi_ABASCAL']['texto_filtrado']))

#Representacion de las palabras más usadas con y sin filtrar
print('\n\nLas frecuencias en el texto sin filtrar son: \n %s' %palabras_rivera2)
print('\n\nLas frecuencias en el texto filtrado son: \n %s' %palabras_rivera)
word=[]
frequency=[]
for i in range(len(palabras_rivera)):
  word.append(palabras_rivera[i][0])
  frequency.append(palabras_rivera[i][1])
indices = np.arange(len(palabras_rivera))
sns.barplot(indices,frequency,palette="Blues_d")
plt.xticks(indices, word, rotation='vertical')
plt.show()

word=[]
frequency=[]
for i in range(len(palabras_rivera2)):
  word.append(palabras_rivera2[i][0])
  frequency.append(palabras_rivera2[i][1])
indices = np.arange(len(palabras_rivera2))
sns.barplot(indices,frequency,palette="Blues_d")
plt.xticks(indices, word, rotation='vertical')
plt.show()

#Organizamos por autores y hacemos las correlaciones
s = (tweets['texto_filtrado'].apply(lambda x: pd.Series(list(x.split())))
                  .stack()
                  .rename('tokens')
                  .reset_index(level=1, drop=True))
df = tweets.join(s).reset_index(drop=True)
df=df[['autor','tokens','fecha']]
z=df.groupby(['autor','tokens']).count()

a=pd.pivot_table(z,columns=['autor'],index=['tokens'])['fecha']
a[np.isnan(a)]=0

#Pinturas y contrastes
print(a.corr(method='pearson'))
#Comprobamos matriz de autocorrelaciones con el p-valor de pearson
pearsonr(a['Albert_Rivera'],a['Albert_Rivera'])

pd.scatter_matrix(a)
plt.show()

plt.matshow(a.corr())
plt.xticks(range(len(a.columns)), a.columns)
plt.yticks(range(len(a.columns)), a.columns)
plt.colorbar()
#plt.show()

#Graficas por separado
p1=a.plot(kind='scatter',x='Albert_Rivera',y='Pablo_Iglesias_')
for i in range(len(a.index)):
    x=a['Albert_Rivera'][i]
    y=a['Pablo_Iglesias_'][i]
    if x>100 or y>100:
        p1.text(x,y,a.index[i])
plt.show()

#odd ratios entre los mensajes publicados por cada uno de los autores
s = (tweets['texto_filtrado'].apply(lambda x: pd.Series(list(x.split())))
                  .stack()
                  .rename('tokens')
                  .reset_index(level=1, drop=True))
df = tweets.join(s).reset_index(drop=True)
df=df[['tokens','autor','fecha']]
z=df.groupby(['autor','tokens']).count()
z1=pd.pivot_table(z,columns=['autor'],index=['tokens'])['fecha']
z1[np.isnan(z1)]=0
for nombre in z1.columns:
    z1['%s_odd' % nombre]=z1[nombre]/(z1[nombre].count()-z1[nombre])
for nombre in z1.columns:
    for nombre2 in z1.columns:
        if 'odd' in nombre and 'odd' in nombre2 and nombre!=nombre2 and 'ratio' not in nombre and 'ratio' not in nombre2:
            z1['%s_%s_ratio' % (nombre,nombre2)]=z1[nombre]
            for i in range(len(z1[nombre])):
                if z1[nombre2][i]>0 and z1[nombre][i]>0:
                    z1['%s_%s_ratio' % (nombre,nombre2)][i]=np.log(z1[nombre][i]/z1[nombre2][i]) #comprobar si falta un logaritmo
                else:
                    z1['%s_%s_ratio' % (nombre,nombre2)][i]=0

rivera_casado=z1[z1['Albert_Rivera']+z1['pablocasado_']>20]['Albert_Rivera_odd_pablocasado__odd_ratio']
rivera_casado=rivera_casado[rivera_casado.abs()>2]
rivera_casado=rivera_casado.sort_values()
rivera_casado.plot(kind='barh')
plt.xlabel('logs odd ratio')
plt.show()

#---------Analisis de sentimiento. Se cargan un conjunto de palabras previamente clasificadas y se mira el numero de veces que cada una de ellas aparece en los mensajes
pos_words=pd.read_csv('.../Desktop/Big data/isol/positivas_mejorada.csv',sep='\t',engine='python',header=None)
neg_words=pd.read_csv('.../Desktop/Big data/isol/negativas_mejorada.csv',sep='\t',engine='python',header=None)

def analisis_sentimiento(texto,pos_word,neg_word):
    sentimiento=[]
    for i in range(len(texto)):
        lista_palabras=[word for word in texto[i].split()]
        match_neg=0
        match_pos=0
        for palabra in neg_word[0]:
            if palabra in lista_palabras:
                match_neg=match_neg+1
        for palabra in pos_word[0]:
            if palabra in lista_palabras:
                match_pos=match_pos+1
        score=match_pos-match_neg
        sentimiento.append(score)
    return sentimiento

tweets['sentimiento']=analisis_sentimiento(tweets['texto_filtrado'],pos_words,neg_words)

#grafico de las medias de sentimiento en general en la red social
b=tweets.groupby(['autor']).sum()
b['numero']=tweets.groupby('autor').count()['fecha']
b['media']=b['sentimiento']/b['numero']
colors=['orange','purple','green','blue','red']
b['media'].plot(kind='bar',color=colors,x='autor')
plt.ylabel('Sentimiento medio')
plt.show()

#grafico por autores de la media de sentimiento
a=tweets.groupby(['autor','sentimiento']).count()['fecha'].unstack()
a[np.isnan(a)]=0
a=a.div(a.sum(axis=1)/100,axis=0).reset_index()

for nombre in a['autor']:
    a[a['autor']==nombre].plot(kind='bar',xlim=[-6,6],cmap="BuPu")
    plt.title(nombre)
plt.show()

#se suma los mensajes positivos y los negativos
a['negativo']=a[-7]+a[-6]+a[-5]+a[-4]+a[-3]+a[-2]+a[-1]
a['positivo']=a[8]+a[7]+a[6]+a[5]+a[4]+a[3]+a[2]+a[1]
a=a.sort_values(by='negativo')
a[['negativo',0,'positivo']].plot.barh(stacked=True)
plt.show()

#Evolucion del sentimiento en el tiempo
c=tweets.groupby(['mes_dia']).sum()
c['numero']=tweets.groupby(['mes_dia']).count()['fecha']
c['media']=c['sentimiento']/c['numero']
c=c.reset_index()
c['mes_dia'] = pd.to_datetime(c['mes_dia'])
pd.Series(data=c['media'].values,index=c['mes_dia']).plot(x='mes_dia',linewidth=0.5)

#linea de regresion del sentimiento
c.insert(c.shape[1],'row_count',c.index.value_counts().sort_index().cumsum())
plt.scatter(c[c['media']>0]['row_count'],c[c['media']>0]['media'],color='green')
plt.scatter(c[c['media']<0]['row_count'],c[c['media']<0]['media'],color='red')
fig=sns.regplot(data=c,x=c['row_count'],y=c['media'],scatter=False)
fig.set_xticklabels(c['label'])
plt.show()

#Tendencia por autor
c=tweets.groupby(['autor','mes_dia']).sum()
c['numero']=tweets.groupby(['autor','mes_dia']).count()['fecha']
c['media']=c['sentimiento']/c['numero']
c=c.reset_index()
c['label']=c['mes_dia']
c['mes_dia'] = pd.to_datetime(c['mes_dia'])

c.insert(c.shape[1],'row_count',c.index.value_counts().sort_index().cumsum())
fig=sns.lmplot(data=c,x='row_count',y='media',hue='autor',scatter=False,palette=('orange','purple','green','blue','red'))
fig.set_xticklabels(c['label'])
plt.show()

c=tweets
c['sentimiento']=[x if x>=0 else -1 for x in c['sentimiento']]
c['sentimiento']=[x if x<=0 else 1 for x in c['sentimiento']]
d=c.groupby(['mes_dia','sentimiento']).count()
d['numero']=c.groupby(['mes_dia','sentimiento']).count()['fecha']
d=d.reset_index() 
d['sentimiento_normalizado']=d.groupby('mes_dia')['numero'].apply(lambda x: x/x.sum())
d['mes_dia'] = pd.to_datetime(d['mes_dia'])

#pd.Series(data=d[d['sentimiento']==1]['sentimiento_normalizado'].values,index=d[d['sentimiento']==1]['mes_dia']).plot(x='mes_dia',color='g',linewidth=0.05,style='.-')
#pd.Series(data=d[d['sentimiento']==-1]['sentimiento_normalizado'].values,index=d[d['sentimiento']==-1]['mes_dia']).plot(x='mes_dia',color='r',linewidth=0.05,style='.-')
plt.show()
for sentimiento in [-1,1]:
    pd.Series(data=d[d['sentimiento']==sentimiento]['numero'].values,index=d[d['sentimiento']==sentimiento]['mes_dia']).plot(x='mes_dia',drawstyle="steps",linewidth=0.7)
plt.show()
plt.legend(['Sentimiento negativo','Sentimiento positivo'])

#---------- Analisis de las emociones
emociones=pd.read_csv('/Users/raulfernandez/Desktop/Big data/emociones.txt',sep='\t',engine='python')
def analisis_emociones(texto,emociones):
    texto=texto.reset_index()
    enfado=0
    anticipacion=0
    asco=0
    miedo=0
    alegria=0
    tristeza=0
    sorpresa=0
    confianza=0
    texto=texto.reset_index()
    for i in range(len(texto)):
        lista_palabras=[word for word in texto['texto_filtrado'][i].split()]
        total=len(texto)
        for palabra in emociones[emociones['Anger']==1]['Palabras']:
            if palabra in lista_palabras:
                enfado=enfado+1
        for palabra in emociones[emociones['Anticipation']==1]['Palabras']:
            if palabra in lista_palabras:
                anticipacion=anticipacion+1
        for palabra in emociones[emociones['Disgust']==1]['Palabras']:
            if palabra in lista_palabras:
                asco=asco+1
        for palabra in emociones[emociones['Fear']==1]['Palabras']:
            if palabra in lista_palabras:
                miedo=miedo+1
        for palabra in emociones[emociones['Joy']==1]['Palabras']:
            if palabra in lista_palabras:
                alegria=alegria+1
        for palabra in emociones[emociones['Sadness']==1]['Palabras']:
            if palabra in lista_palabras:
                tristeza=tristeza+1
        for palabra in emociones[emociones['Surprise']==1]['Palabras']:
            if palabra in lista_palabras:
                sorpresa=sorpresa+1
        for palabra in emociones[emociones['Trust']==1]['Palabras']:
            if palabra in lista_palabras:
                confianza=confianza+1
        total=enfado+anticipacion+asco+miedo+alegria+tristeza+sorpresa+confianza
    return enfado/total,anticipacion/total,asco/total,miedo/total,alegria/total,tristeza/total,sorpresa/total,confianza/total
my_colors = 'rgbkymc'
ax=sns.barplot(np.arange(8),analisis_emociones(tweets[tweets['autor']=='Albert_Rivera']['texto_filtrado'],emociones))
plt.xticks(np.arange(8),('enfado','anticipacion','asco','miedo','alegria','tristeza','sorpresa','confianza'))
ax.set(ylabel='Procentaje')
plt.title('Albert Rivera')
plt.legend()
plt.show()

#numero de rts
rts=tweets.groupby(['autor','mes_dia']).sum()['retweet_count'].reset_index()
rts['mes_dia'] = pd.to_datetime(rts['mes_dia'])
rts['numero']=tweets.groupby(['autor','mes_dia']).count().reset_index()['fecha']
rts['rts_por_tweet']=rts['retweet_count']/rts['numero']
rts_rivera=pd.Series(data=rts[rts['autor']=='Albert_Rivera']['rts_por_tweet'].values,index=rts[rts['autor']=='Albert_Rivera']['mes_dia'])
ax=rts_rivera.plot(color='orange',x='mes_dia')
rts_iglesias=pd.Series(data=rts[rts['autor']=='Pablo_Iglesias_']['rts_por_tweet'].values,index=rts[rts['autor']=='Pablo_Iglesias_']['mes_dia'])
rts_iglesias.plot(ax=ax,color='purple',x='mes_dia')
rts_casado=pd.Series(data=rts[rts['autor']=='pablocasado_']['rts_por_tweet'].values,index=rts[rts['autor']=='pablocasado_']['mes_dia'])
rts_casado.plot(ax=ax,color='b',x='mes_dia')
rts_sanchez=pd.Series(data=rts[rts['autor']=='sanchezcastejon']['rts_por_tweet'].values,index=rts[rts['autor']=='sanchezcastejon']['mes_dia'])
rts_sanchez.plot(ax=ax,color='r',x='mes_dia')
rts_abascal=pd.Series(data=rts[rts['autor']=='Santi_ABASCAL']['rts_por_tweet'].values,index=rts[rts['autor']=='Santi_ABASCAL']['mes_dia'])
rts_abascal.plot(color='g')
plt.show()

key_events = [(pd.to_datetime('Jan 23 2019'), 'Conflicto Venezuela'),
              (pd.to_datetime('Feb 27 2018'), 'Mobile World Congress'),
              (pd.to_datetime('Jun 11 2018'), 'Barco Aquarius'),
              (pd.to_datetime('Aug 09 2018'), 'Reunión Sanchez-Torra'),
              (pd.to_datetime('Oct 12 2018'), '12 Octubre'),
              (pd.to_datetime('Dec 20 2018'), 'Pacto Andalucia'),
              ]

for event in key_events:
    plt.axvline(event[0], color='black',linewidth=0.5,linestyle='--')
    plt.text(event[0] + pd.Timedelta(1, 'm'), 15000, event[1], rotation=90, size=7)
plt.show()

rts.boxplot(column='rts_por_tweet',by='autor')
plt.show()


#numero de favoritos
fav=tweets.groupby(['autor','mes_dia']).sum()['favorite_count'].reset_index()
fav['mes_dia'] = pd.to_datetime(fav['mes_dia'])
fav['numero']=tweets.groupby(['autor','mes_dia']).count().reset_index()['fecha']
fav['fvs_por_tweet']=fav['favorite_count']/fav['numero']
fav_rivera=pd.Series(data=fav[fav['autor']=='Albert_Rivera']['fvs_por_tweet'].values,index=fav[fav['autor']=='Albert_Rivera']['mes_dia'])
ax=fav_rivera.plot(color='orange',x='mes_dia')
fav_iglesias=pd.Series(data=fav[fav['autor']=='Pablo_Iglesias_']['fvs_por_tweet'].values,index=fav[fav['autor']=='Pablo_Iglesias_']['mes_dia'])
fav_iglesias.plot(ax=ax,color='purple',x='mes_dia')
fav_casado=pd.Series(data=fav[fav['autor']=='pablocasado_']['fvs_por_tweet'].values,index=fav[fav['autor']=='pablocasado_']['mes_dia'])
fav_casado.plot(ax=ax,color='b',x='mes_dia')
fav_sanchez=pd.Series(data=fav[fav['autor']=='sanchezcastejon']['fvs_por_tweet'].values,index=fav[fav['autor']=='sanchezcastejon']['mes_dia'])
fav_sanchez.plot(ax=ax,color='r',x='mes_dia')
fav_abascal=pd.Series(data=fav[fav['autor']=='Santi_ABASCAL']['fvs_por_tweet'].values,index=fav[fav['autor']=='Santi_ABASCAL']['mes_dia'])
fav_abascal.plot(color='g')
plt.show()

#Wordcloud por sentimiento
a=tweets[tweets['autor']=='Albert_Rivera']['texto_filtrado']
a=pd.DataFrame(tweets[tweets['autor']=='Albert_Rivera']['texto_filtrado'])
a=a.texto_filtrado.str.split(expand=True).stack().value_counts().reset_index()
b=pd.DataFrame(a)
b.rename(columns={'index':'Palabras',0:'frecuencia'},inplace=True)

emociones=pd.read_csv('/Users/raulfernandez/Desktop/Big data/emociones.txt',sep='\t',engine='python')
print(b)
frame=pd.merge(b,emociones,on='Palabras',how='inner')
frame['enfado']=frame['frecuencia']*frame['Anger']
frame['anticipacion']=frame['frecuencia']*frame['Anticipation']
frame['asco']=frame['frecuencia']*frame['Disgust']
frame['miedo']=frame['frecuencia']*frame['Fear']
frame['alegria']=frame['frecuencia']*frame['Joy']
frame['tristeza']=frame['frecuencia']*frame['Sadness']
frame['sorpresa']=frame['frecuencia']*frame['Surprise']
frame['confianza']=frame['frecuencia']*frame['Trust']
frame.to_csv('/Users/raulfernandez/Desktop/Big data/frame.csv')

#Hashtags
def hashtag(texto):
    hashtag=[]
    for i in texto:
        hashtag.append(re.findall(r'#(\w+)',i))
    return sum(hashtag,[])

hashtag_rivera=hashtag(tweets[tweets['autor']=='Albert_Rivera']['texto'])
hashtag_iglesias=hashtag(tweets[tweets['autor']=='Pablo_Iglesias_']['texto'])
hashtag_casado=hashtag(tweets[tweets['autor']=='pablocasado_']['texto'])
hashtag_sanchez=hashtag(tweets[tweets['autor']=='sanchezcastejon']['texto'])
hashtag_abascal=hashtag(tweets[tweets['autor']=='Santi_ABASCAL']['texto'])

df1=pd.DataFrame({'Hashtag':list(collections.Counter(hashtag_rivera).keys()),'rivera':list(collections.Counter(hashtag_rivera).values())})
df2=pd.DataFrame({'Hashtag':list(collections.Counter(hashtag_iglesias).keys()),'iglesias':list(collections.Counter(hashtag_iglesias).values())})
df1=pd.merge(df1,df2,on='Hashtag',how='outer')
df3=pd.DataFrame({'Hashtag':list(collections.Counter(hashtag_casado).keys()),'casado':list(collections.Counter(hashtag_casado).values())})
df1=pd.merge(df1,df3,on='Hashtag',how='outer')
df4=pd.DataFrame({'Hashtag':list(collections.Counter(hashtag_sanchez).keys()),'sanchez':list(collections.Counter(hashtag_sanchez).values())})
df1=pd.merge(df1,df4,on='Hashtag',how='outer')
df5=pd.DataFrame({'Hashtag':list(collections.Counter(hashtag_abascal).keys()),'abascal':list(collections.Counter(hashtag_abascal).values())})
df1=pd.merge(df1,df5,on='Hashtag',how='outer')
df1=df1.fillna(0)

ax=sns.barplot(data=df1.nlargest(20,'rivera'),x='Hashtag',y='rivera',label='tiny')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
plt.ylabel('Numero de Hashtag')
plt.title('Albert Rivera')
plt.show()

#Modelos de machine learning ----------------

#Previamente se crea una matriz binaria que indica la aparición de una palabra en un mensaje

s = (tweets['texto_filtrado'].apply(lambda x: pd.Series(list(x.split())))
                  .stack()
                  .rename('tokens')
                  .reset_index(level=1, drop=True))
df = tweets.join(s).reset_index(drop=True)
df1=df[['fecha','tokens']]
z=df1.groupby(['tokens']).count()['fecha'].reset_index()
z1=z.sum()
z.rename(columns={'fecha':'frecuencia_token'},inplace=True)
z['tf']=z['frecuencia_token']/z1['fecha']
z2=df.groupby(['tokens']).nunique()['texto_filtrado'].reset_index()
z['numero_tweets']=tweets.count()['fecha']
z['veces_repetido']=z2['texto_filtrado']
z['idf']=np.log(z['numero_tweets']/z['veces_repetido'])
z['tfidf']=z['tf']*z['idf']

ax=sns.barplot(data=z.nlargest(20,'tfidf'),x='tokens',y='tfidf',label='tiny',palette="GnBu_d")
ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
plt.show()

####### Modelos
#se convierten los mensajes en vectores, pasando las palabras a valores numéricos
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
text_tf= vectorizer.fit_transform(tweets['texto'])

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(text_tf, tweets['sentimiento'], test_size=0.5, random_state=1)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

pred=clf.predict(X_test)
clases=['-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7']
cm=confusion_matrix(y_test,pred)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=clases, yticklabels=clases,
           ylabel='True label', xlabel='Predicted label')
plt.show()

print(classification_report(y_test,pred))

texto=['Los grandes medios no solo manipulan y mienten sobre VOX. También mienten en cualquiera de sus noticias. Estos salvajes deben salir de España y no volver nunca más una vez acabada su ridícula condena. Pero: ¿por qué se ha silenciado este horror?',
'Lamento profundamente la muerte del joven Marcos Garrido en Jerez. Catorce años no es edad para morir. Toda una vida por delante truncada. Me sumo al terrible dolor de sus padres, familiares y amigos',
'Enhorabuena a las Leonas que se han proclamado heptacampeonas de @rugby_europe y han batido el récord de asistencia a un partido de rugby en España 🏉  Un nuevo éxito compartido que revalida el espíritu de equipo, trabajo y tesón que os caracteriza.']
for i in range(len(texto)):
    texto[i]=tokenizar(texto[i])
print(texto)
test_df=vectorizer.transform(texto)
print (clf.predict(test_df))
 
#Modelo de los k-vecinos 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
text_tf= vectorizer.fit_transform(tweets['texto'])

from sklearn.model_selection import train_test_split

tweets[tweets['sentimiento']>0]=1
tweets[tweets['sentimiento']<0]=2

X_train, X_test, y_train, y_test = train_test_split(text_tf, tweets['sentimiento'], test_size=0.3, random_state=123)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

neutro=clf.predict_proba(text_tf)[:,0]
positivo=clf.predict_proba(text_tf)[:,1]
negativo=clf.predict_proba(text_tf)[:,2]
plt.subplot(212)
plt.hist(neutro[neutro>0])
plt.subplot(221)
plt.hist(positivo[positivo>0])
plt.subplot(222)
plt.hist(negativo[negativo>0])
plt.show()

#Modelo randomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

pred=clf.predict(X_test)
clases=['0','1','2']
cm=confusion_matrix(y_test,pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=clases, yticklabels=clases,
           ylabel='True label', xlabel='Predicted label')
plt.show()

print(classification_report(y_test,pred))


#Modelo predictivo para determinar el autor del texto
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(ngram_range=(1,2))
text_tf= vectorizer.fit_transform(tweets['texto'])


def conversor(sentiment):
    return {
        'Albert_Rivera': 0,
        'Pablo_Iglesias_': 1,
        'pablocasado_':2,
        'sanchezcastejon' : 3,
        'Santi_ABASCAL' :4,
    }[sentiment]
tweets_autores = tweets.autor.apply(conversor)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_tf, tweets_autores, test_size=0.4, random_state=0)


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
print('hola')
clf.fit(X_train, y_train)
print('hola')
print(clf.score(X_test, y_test))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pred=clf.predict(X_test)
clases=['Rivera','Iglesias','Casado','Sanchez','Abascal']
cm=confusion_matrix(y_test,pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=clases, yticklabels=clases,
           ylabel='True label', xlabel='Predicted label')
plt.show()

print(classification_report(y_test,pred))

sentences = vectorizer.transform([
    "Es una emergencia nacional enviar a Sánchez y a sus socios separatistas a la oposición. Tiendo la mano a Casado para formar un Gobierno Ciudadanos-PP tras las elecciones del 28-A y abrir una nueva etapa en nuestro país. Que los españoles elijan quién lo preside. #GaliciaNaranja",
    "¿Qué pasa cuando la banca tiene acciones de un medio de comunicación? Pues está demostrado que LaBancaManda, anteponiendo sus intereses políticos y económicos a tu derecho a la información y a los derechos de los periodistas.¡Vamos a darle la vuelta a esto!",
    "Es lamentable que la Junta Electoral tenga que hacer el trabajo del Gobierno. Sánchez no hace nada ante órdenes ilegales a los Mossos, ni la retirada de lazos amarillos, ni la propaganda de TV3. Es insólito que por los votos de los separatistas no haga cumplir la ley.",
    "De los gobiernos depende que se corrijan lacras inasumibles por una sociedad que quiera ser justa de verdad. Debemos acabar con situaciones injustificables como la brecha salarial y terminar con los techos de cristal para alcanzar una #igualdad real y efectiva. #MakeHERStory",
    "La #EspañaViva abarrota Santander. Un tsunami de ilusión y esperanza barrerá el miedo y la mentira de los medios,  de los partidos y de los tezanos de izquierdas y derechas. #SantanderPorEspaña #Santander",
    "De nuevo con @JcQuer, cuya lucha compartimos tantos españoles. Me comprometo a mantener la prisión permanente revisable, respetar lo que diga el TC, ampliar supuestos para aplicarla y luchar contra la reincidencia por el cumplimiento íntegro de las penas. Siempre con las víctimas",
    "En la #EspañaVaciada no tienen fácil acceso a un pediatra, a una escuela o a un transporte público. Quiero un país con las mismas oportunidades y derechos para todos, vivan en el pueblo o en la ciudad. Lideraremos un pacto de Estado contra la despoblación desde el nuevo Gobierno",
    "Las maniobras de policías corruptos y ciertos periodistas para que Unidas Podemos no entrase a formar parte de un Gobierno no nos afectan solo a nosotros, son algo grave que pone en riesgo la calidad de la democracia española. Algunos medios deberían disculparse ante su público",
    "Los que ni pudisteis, ni podéis, ni podréis nunca con España sois vosotros. Ni con tiros en la nuca, ni con bombas lapa, ni con secuestros ni con escaños ilegítimos. Ni secuestrando el Congreso ni volándolo. Ni matándonos a todos. Nunca podréis. ¡VIVA ESPAÑA!",
    "No es creíble la rectificación de Iceta sobre el independentismo. Ha sido muy claro, avala la estrategia de Sánchez no solo de vender España a trozos sino de venderla a plazos. Lo mismo que Sánchez negociaba en Pedralbes 👉 Es una traición a España a la que Iceta pone plazo"
])
print(clf.predict_proba(sentences))

#SEGUIDORES. COINCIDENCIAS POR AUTOR
rivera = pd.read_csv('/Users/raulfernandez/Desktop/Big data/seguidores_rivera.csv')
casado = pd.read_csv('/Users/raulfernandez/Desktop/Big data/seguidores_casado.csv')
sanchez = pd.read_csv('/Users/raulfernandez/Desktop/Big data/seguidores_sanchez.csv')
iglesias = pd.read_csv('/Users/raulfernandez/Desktop/Big data/seguidores_iglesias.csv')
abascal = pd.read_csv('/Users/raulfernandez/Desktop/Big data/seguidores_abascal.csv')

print(rivera.describe())

nombres=[casado,sanchez,rivera,iglesias,abascal]
a=[]
for nombre in nombres:
    for nombre2 in nombres:
        a.append(len(nombre.merge(nombre2,on='user_id',how='inner'))/len(nombre))
a=np.reshape(a,(5,5))
nombres=['casado','sanchez','rivera','iglesias','abascal']
plt.matshow(a)
plt.xticks(range(len(nombres)),nombres)
plt.yticks(range(len(nombres)), nombres)
plt.colorbar()
plt.show()

#grafica seguidores
casado = pd.read_csv('/Users/raulfernandez/Desktop/Big data/user_detail_casado.csv')
casado['followers_count'].plot.hist(bins=200,range=[0, 200])
#abascal.plot(logx=True)
plt.show()

labels = ['Seguidores falsos','Seguidores reales']
sizes=[len(abascal[abascal['followers_count']==0])/len(abascal['followers_count'])*100,len(abascal[abascal['followers_count']>0])/len(abascal['followers_count'])*100]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90,colors=['orange','g'])
#ax1.axis('equal')
plt.show()
