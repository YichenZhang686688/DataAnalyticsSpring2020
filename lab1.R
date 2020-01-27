# Creating a dataframe
# Example: RPI Weather dataframe

days <- c('Mon','Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun') # days
temp <- c(28,30.5,32,31.2,29.3,27.9,26.4) # Temperature in F' during the winter
snowed <- c('T','T','F','F','T','T','F') # Snowed on that day:T = TRUE, F= FALSE
help("data.frame")
RPI_Weather_Week <- data.frame(days, temp, snowed) # creating the dataframe using the data.frame

RPI_Weather_Week
head(RPI_Weather_Week) # head of the data frame, NOTE: it will snow only 6 rows, usually head() function shows the 
#first 6 rows of the dataframe, hewe we have onlhy 7 rows in our dataframe.

str(RPI_Weather_Week) # we can take a look at the structure of the dataframe using the str() function

summary(RPI_Weather_Week) # summary of the dataframe using the summary function

RPI_Weather_Week[1,] # showing the 1st row and all columns
RPI_Weather_Week[,1] # showing the 1st column and all rows

RPI_Weather_Week[,'snowed']
RPI_Weather_Week[,'days']
RPI_Weather_Week[,'temp']
RPI_Weather_Week[1:5,c("days","temp")]
RPI_Weather_Week$temp
subset(RPI_Weather_Week, SUBSET = snowed == TRUE)

sorted.snowed <- order(RPI_Weather_Week['snowed'])
sorted.snowed
RPI_Weather_Week[sorted.snowed,]

# RPI_Weather_Week[descending_snowed,]
dec.snow <- order(-RPI_Weather_Week$temp)
dec.snow
#Creating dataframes
#creating an empty dataframe
empty.dataframe <- data.frame()
v1 <- 1:10
v1
letters
v2 <- letters[1:10]
df <- data.frame(col.name.1 = v1, col.name.2 = v2)
df


#Reading
rm(list=ls())
install.packages("readxl")
Yes
library(readxl)
setwd("~/Desktop")
getwd()
EPI_data<-read.csv("~/Desktop/2010EPI_data.csv",header = TRUE)
EPI2010_all_countries <- read_xls("~/Desktop/2010EPI_data.xls", sheet = 5)


EPI_data <- read.csv(file.choose(), header = TRUE)
help("read.csv")
View(EPI_data)
attach(EPI_data)

fix(EPI_data)
EPI
tf<-is.na(EPI)
E<-EPI[!tf]
E
summary(EPI)
fivenum(EPI,na.rm=TRUE)
stem(EPI)
hist(EPI)
hist(EPI,seq(30.,95.,1.0),pro=TRUE)
lines(density(EPI,na.rm=TRUE,bw=1.))
lines(density(EPI,na.rm=TRUE,bw="SJ"))
help(stem)
rug(EPI)

plot(ecdf(EPI), do.points=FALSE, verticals=TRUE) 
par(pty="s")
qqnorm(EPI)
qqline(EPI)
x<-seq(30,95,1)
qqplot(qt(ppoints(250),df=5),x,xlab="Q-Q plot for t dsn")
qqline(x)
dev.off()
boxplot(EPI,DALY) 
qqplot(EPI,DALY)
help(distributions)
