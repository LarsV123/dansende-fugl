# Uncomment and run if not already installed.
#install.packages('jsonlite', dependencies=TRUE, repos='http://cran.rstudio.com/')
#install.packages('RPostgreSQL')
#install.packages('RPostgres')
#install.packages('DBI')
#install.packages('dotenv')
#install.packages('ciTools')
#install.packages('ggplot2')

library(RPostgreSQL)
library(RPostgres)
library(DBI)
library(dotenv)
library(ciTools)
library(ggplot2)
library(ggfortify)

# Connect to database
#getwd() # to check working directory
setwd("~/Desktop/eit/dansende-fugl") # set to your project path
load_dot_env(file = ".env")
db <- Sys.getenv("POSTGRES_DB")
host_db <- Sys.getenv("HOST")
db_user <- Sys.getenv("DB_USER")
db_password <- Sys.getenv("DB_PASSWORD")
con <- dbConnect(RPostgres::Postgres(), dbname = db, host=host_db, user=db_user, password=db_password)  

# Load data
db <- dbGetQuery(con, "SELECT mbti, score, word_count_quoteless FROM unique_no_comments")

################ SCORE ###########################

score.lm <- lm(score ~ mbti, data=db)
summary(score.lm)

d = data.frame("mbti"= unique(db$mbti))
d$fit = (predict(score.lm, newdata=d,se.fit = T))$fit
d$se.fit = (predict(score.lm, newdata=d,se.fit = T))$se.fit
d$lci = d$fit-1.96*d$se.fit
d$uci = d$fit+1.96*d$se.fit
ggplot(d,aes(y=fit,x=mbti,ymin=lci,ymax=uci,label=round(fit,digits=2)))+
  geom_point()+
  geom_errorbar()+
  ylab("Gjennomsnittlig score")+
  xlab("MBTI personlighetstype")+
  geom_text(hjust=1.2,size=4)+
  theme(axis.text = element_text(size = 12), axis.title=element_text(size = 14))+
  coord_cartesian(clip='off')
  