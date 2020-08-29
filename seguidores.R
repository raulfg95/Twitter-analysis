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

#obtenemos permisos
consumer_key <- ""
consumer_secret <- ""
access_token <- ""
access_secret <- ""

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

#para cada persona buscamos los identificadores de todos sus seguidores
seguidores_rivera<-get_followers('Albert_Rivera',n=lookup_users('Albert_Rivera')$followers_count,retryonratelimit=TRUE)
seguidores_iglesias<-get_followers('Pablo_Iglesias_',n=lookup_users('Pablo_Iglesias_')$followers_count,retryonratelimit=TRUE)
seguidores_casado<-get_followers('pablocasado_',n=lookup_users('pablocasado_')$followers_count,retryonratelimit=TRUE)
seguidores_sanchez<-get_followers('sanchezcastejon',n=lookup_users('sanchezcastejon')$followers_count,retryonratelimit=TRUE)
seguidores_abascal<-get_followers('Santi_ABASCAL',n=lookup_users('Santi_ABASCAL')$followers_count,retryonratelimit=TRUE)

#En cada caso, buscamos para cada seguidor el número de seguidores que tiene.
#Primero se configura el id del seguidor para que sea numérico
seguidores<-seguidores_rivera %>% pull(user_id)%>%unique
seguidores<-gsub('x', "", seguidores)
seguidores<-as.numeric(seguidores)
seguidores<-sort(seguidores,decreasing=T)
set.seed(1)
rand<-sample(length(seguidores))
seguidores<-seguidores[rand]

user_details<-lookup_users(seguidores)
user_detail<-user_details %>% arrange(desc(followers_count))%>% select(screen_name, followers_count)
write_as_csv(user_details, "/Users/raulfernandez/Desktop/Big data/user_detail_rivera.csv")
