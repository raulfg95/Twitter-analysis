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

consumer_key <- ""
consumer_secret <- ""
access_token <- ""
access_secret <- ""

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

#Obtenemos los tweets
users<-c('Albert_Rivera','Pablo_Iglesias_','pablocasado_','sanchezcastejon','Santi_ABASCAL')
tweets <- vector("list", length(users))
tweets<-get_timelines(c(users),n = 3200, retryonratelimit =TRUE)


#Nos quedamos con las variables interesantes: autor, fecha de creación, mensaje, número de favoritos, total de retweets, hashtags utilizados y la localización
tweetsold<-tweets
write_as_csv(tweetsold, ".../Desktop/Big data/tweetsold.csv")
tweets <- tweetsold[c("screen_name","created_at","status_id", "text", "favorite_count", "is_retweet","retweet_count","source","hashtags","mentions_screen_name","retweet_screen_name","retweet_location","geo_coords")]
tweets<-rename(tweets,replace=c("screen_name"="autor","created_at"="fecha","text"="texto"))
write_as_csv(tweets, ".../Desktop/Big data/tweets.csv")

#Representamos el número de tweets por mes, por día y mes y por hora
hr <- hour(tweets$fecha)
minu <- minute(tweets$fecha)
sec <- second(tweets$fecha)
hora<-rbind(hr,minu,sec)
tweets_mes_dia <- mutate(tweets,mes_dia = format(fecha, "%Y-%m-%d")) 
tweets_mes_dia2 <- mutate(tweets_mes_dia,mes = format(fecha, "%Y-%m"))
tweets_mes_dia3<- mutate(tweets_mes_dia2,hora=format(fecha,"%H"))
write_as_csv(tweets_mes_dia3, "/Users/raulfernandez/Desktop/Big data/tweets.csv")
tweets_mes_dia %>%group_by(autor,mes_dia)%>% dplyr::summarise(n = n()) %>%
  ggplot(aes(x = mes_dia, y = n,color=autor))+
  geom_line(aes(group = autor)) 
