library("ROAuth");
library("base64enc");
library("twitteR");
library("streamR");
library(ggplot2)
library(dplyr)
library(plyr)
library(stringr)
library(lubridate)
library(tidyverse)
library(rtweet)
library(tidyverse)
library(knitr)
library(readr)
library(magrittr)
library(wordcloud)
library(tm)


tweets <- read_csv(".../Desktop/Big data/tweetsold.csv")

#Se realiza una nube de palabras agrupadas en función del sentimiento
emociones<-read.csv('.../Desktop/Big data/frame.csv',header=TRUE,sep=",")[ ,c('Palabras','enfado','anticipacion','asco','miedo','alegria','tristeza','sorpresa','confianza')]
emociones=emociones[!duplicated(emociones$Palabras), ]
df1 <- data.frame(emociones[,-1], row.names = emociones[,1])
df1<-df1[order(df1$enfado),]
comparison.cloud(df1,title.size=1,random.order=FALSE,colors = c("#00B2FF", "red", "#FF0099", "#6600CC", "green", "orange", "blue", "brown"), max.words=2000,scale=c(1.5, 0.4),rot.per=0.4)
