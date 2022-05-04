library(RPostgreSQL)
library(RPostgres)
library(DBI)
library(dotenv)
library(ciTools)
library(ggplot2)
library(ggfortify)
library(ggpubr)

# Connect to database
#getwd() # to check working directory
setwd("~/Desktop/eit/dansende-fugl") # set to your project path
load_dot_env(file = ".env")
db <- Sys.getenv("POSTGRES_DB")
host_db <- Sys.getenv("HOST")
db_user <- Sys.getenv("DB_USER")
db_password <- Sys.getenv("DB_PASSWORD")
con <- dbConnect(RPostgres::Postgres(), dbname = db, host=host_db, user=db_user, password=db_password)  

# INFJ
db.infj <- dbGetQuery(con, "SELECT * FROM hour_confint_infj")
db.infj <- as.data.frame(db.infj)

ggplot(db.infj,aes(y=p,x=hour,ymin=lci,ymax=uci))+
  geom_point()+
  geom_ribbon(alpha=0.4)+
  ylab("Andel kommentarer")+
  xlab("Tidspunkt på døgnet")+
  coord_cartesian(clip='off')

# IE
db.ie <- as.data.frame(dbGetQuery(con, "SELECT * FROM hour_confint_ie"))
ggplot(db.ie,aes(y=p,x=hour,ymin=lci,ymax=uci, color=ie))+
  geom_point()+
  geom_ribbon(alpha=0.4)+
  ylab("Andel kommentarer")+
  xlab("Tidspunkt på døgnet")+
  coord_cartesian(clip='off')

# SN
db.sn <- as.data.frame(dbGetQuery(con, "SELECT * FROM hour_confint_sn"))
ggplot(db.sn,aes(y=p,x=hour,ymin=lci,ymax=uci, color=sn))+
  geom_point()+
  geom_ribbon(alpha=0.4)+
  ylab("Andel kommentarer")+
  xlab("Tidspunkt på døgnet")+
  coord_cartesian(clip='off')

# TF
db.tf <- as.data.frame(dbGetQuery(con, "SELECT * FROM hour_confint_tf"))
ggplot(db.tf,aes(y=p,x=hour,ymin=lci,ymax=uci, color=tf))+
  geom_point()+
  geom_ribbon(alpha=0.4)+
  ylab("Andel kommentarer")+
  xlab("Tidspunkt på døgnet")+
  coord_cartesian(clip='off')

# JP
db.jp <- as.data.frame(dbGetQuery(con, "SELECT * FROM hour_confint_jp"))
ggplot(db.jp,aes(y=p,x=hour,ymin=lci,ymax=uci, color=jp))+
  geom_point()+
  geom_ribbon(alpha=0.4)+
  ylab("Andel kommentarer")+
  xlab("Tidspunkt på døgnet")+
  coord_cartesian(clip='off')
