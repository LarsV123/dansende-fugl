# Uncomment and run if not already installed.
#install.packages('jsonlite', dependencies=TRUE, repos='http://cran.rstudio.com/')
#install.packages('RPostgreSQL')
#install.packages('RPostgres')
#install.packages('DBI')
#install.packages('dotenv')

library(RPostgreSQL)
library(RPostgres)
library(DBI)
library(dotenv)

#getwd() # to check working directory
setwd("~/Desktop/eit/dansende-fugl") # set to your project path
load_dot_env(file = ".env")
db <- Sys.getenv("POSTGRES_DB")
host_db <- Sys.getenv("HOST")
db_user <- Sys.getenv("DB_USER")
db_password <- Sys.getenv("DB_PASSWORD")
con <- dbConnect(RPostgres::Postgres(), dbname = db, host=host_db, user=db_user, password=db_password)  

# SQL query example
dbGetQuery(con, "SELECT COUNT(*) FROM typed_posts")
