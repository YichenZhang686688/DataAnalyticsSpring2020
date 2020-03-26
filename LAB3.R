#lab3

library(rpart)
library(rpart.plot)
data("msleep")
str(msleep)
help("msleep") # read the documentation for the msleep dataset. # It is about mammal’s sleep dataset
# observe the structure of the #msleep dataset
str(data)
# creating a new data frame with the following columns included.
mSleepDF1 <- msleep[,c(3,6,10,11)] # 3 = vore ,6=sleep_total, 10=brainwt, 11=bodywt
# observe the structure of the mSleepDF
str(mSleepDF1)
head(mSleepDF1)
# Building Regression Decision Tree that #predicts the total sleeping
# hours of the mamals based on the other #variables available on the dataset
help("rpart") # Read the documentation for the rpart() function.
sleepModel_1 <- rpart(sleep_total ~ ., data=mSleepDF1, method = "anova")
# method we are using here is anova becuase our target here is sleep_total is a numerical one.
sleepModel_1
# let's visualize this using rpart.plot()
help("rpart.plot")
rpart.plot(sleepModel_1, type = 3, fallen.leaves = TRUE)
rpart.plot(sleepModel_1, type = 3, fallen.leaves = FALSE)
rpart.plot(sleepModel_1, type = 3,digits = 3, fallen.leaves = TRUE) # with 3 digits
rpart.plot(sleepModel_1, type = 3,digits = 4, fallen.leaves = TRUE)
# type = 3, Draw separate split labels for the left and right directions.See the documentation
#fallen.leaves = TRUE, Default TRUE to position the leaf nodes at the bottom of the graph.
#It can be helpful to use FALSE if the graph is too crowded and the text size is too small. 8 rpart.plot(sleepModel_1, type = 3,digits = 3, fallen.leaves = TRUE) # with 3 digits

#Ctree
# install the C50 package 
install.packages("C50")
# require(C50) 
library(C50)
# we will be using the iris dataset to do a #classfication
data("iris")
head(iris) # head of the iris dataset
str(iris) # look at the structure of the dataset using str()
table(iris$Species) # using table() function we can look at the Species of Iris dataset column
# set the seed 
set.seed(9850)
# generate random numbers 
grn <-runif(nrow(iris))
# creating a randomized iris dataset , shuffling the dataset
# we use the order() function along with the #random numbers we generated. 
irisrand <-iris[order(grn),]
# obsrve that rows are now randomly shuffled.
str(irisrand)
classificationmodel1 <-C5.0(irisrand[1:100,-5], irisrand[1:100,5]) 
classificationmodel1
summary(classificationmodel1)
# now we will do the prediction using the #predict() function
# We are using the remaining last 50 rows for #here starting from 101 row to 150th row
prediction1 <- predict(classificationmodel1,irisrand[101:150,])
prediction1
# we will use the confusion matrix to #understand our prediction
# Read the documentation for the table() function in RStudio help 
table(irisrand[101:150,5],prediction1)
# you can write the same above line by defining what is the "predicted" 
## table(irisrand[101:150,5],Predicted = prediction1)
plot(classificationmodel1)


library("e1071") 
classifier<-naiveBayes(iris[,1:4], iris[,5])
table(predict(classifier, iris[,-5]), iris[,5], dnn=list('predicted','actual'))
classifier$apriori 
classifier$tables$Petal.Length 
Standard Deviation
mean
plot(function(x) dnorm(x, 1.462, 0.1736640), 0, 8, col="red", main="Petal length distribution for the 3 different species")
curve(dnorm(x, 4.260, 0.4699110), add=TRUE, col="blue")
curve(dnorm(x, 5.552, 0.5518947 ), add=TRUE, col = "green")


# Heatmap, image, and hierarchical clustering
set.seed(12345)
help(par)
par(mar = rep(0.2, 4))

data_Matrix <- matrix(rnorm(400), nrow = 40)
image(1:10, 1:40, t(data_Matrix)[, nrow(data_Matrix):1])

par(mar = rep(0.2, 4))
heatmap(data_Matrix)
help("rbinom")
set.seed(678910)
for(i in 1:40){
  coin_Flip <- rbinom(1, size = 1, prob = 0.5)
  if(coin_Flip){
  data_Matrix[i, ] <- data_Matrix[i, ] + rep(c(0,3), each =5) }
}


par(mar = rep(0.2, 4))
image(1:10, 1:40, t(data_Matrix)[, nrow(data_Matrix):1])


par(mar = rep(0.2, 4))
heatmap(data_Matrix)


hh <- hclust(dist(data_Matrix))
data_Matrix_ordered <- data_Matrix[hh$order, ]
# plot
par(mfrow = c(1, 3))
image(t(data_Matrix_ordered)[, nrow(data_Matrix_ordered):1])
plot(rowMeans(data_Matrix_ordered), 40:1, xlab = "The Row Mean", ylab = "Row", pch = 19)
plot(colMeans(data_Matrix_ordered), xlab = "Column", ylab = "The Column Mean", pch = 19)


#data(Titanic) rpart, ctree, hclust, randomForest for: Survived ~ .
data("Titanic")
# read the documentation
help("Titanic") 
# observe the structure of the dataset
str(Titanic)
View(Titanic)
head(Titanic)

#rpart
help("rpart") # Read the documentation for the rpart() function.
TitanicModel_1 <- rpart(Survived ~ ., data=Titanic, method = "class")
# method we are using here is anova becuase our target here is Survived is a a factor.
TitanicModel_1
# let's visualize this using rpart.plot()
help("rpart.plot")
rpart.plot(TitanicModel_1, type = 3, fallen.leaves = TRUE)
rpart.plot(TitanicModel_1, type = 3, fallen.leaves = FALSE)

#ctree
install.packages("titanic")
library(titanic)
dim(titanic_train)
head(titanic_train)
dim(titanic_test)
View(titanic_train)
titanic_trainn <- titanic_train[,-c(1,4,9)] 
titanic_trainn
install.packages("C50")
library(C50)
#classfication
# set the seed 
set.seed(9850)
install.packages("psych")
library(psych)
titanic_trainn $ Survived [titanic_trainn $ Survived == 1 ] ="S"
titanic_trainn $ Survived [titanic_trainn $ Survived == 0] ="D"

titanic_trainn$Pclass <- as.character(titanic_trainn$Pclass)
titanic_trainn$SibSp <- as.character(titanic_trainn$SibSp)
titanic_trainn$Parch <- as.character(titanic_trainn$Parch)
titanic_trainn$Age <-as.numeric(titanic_trainn$Age)
titanic_trainne<-is.na(titanic_trainn)
titanic_trainn$Age <-cut(titanic_trainn$Age , br=(2,18,50,71),labels=("childern","Adult","old"))
titanic_trainn$Age <- cut(titanic_trainn$Age, breaks=(2,18,49,71),labels=("childern","Adult","old"))
titanic_trainn <-as.data.frame.factor(titanic_trainn)
titanic_trainn <- titanic_train[,-c(10)] 
titanic_trainn
classificationmodel <-C5.0(titanic_trainn[,-1], titanic_trainn[,1]) 
classificationmodel
summary(classificationmodel1)
# now we will do the prediction using the #predict() function
# We are using the remaining last 50 rows for #here starting from 101 row to 150th row
prediction1 <- predict(classificationmodel1,irisrand[101:150,])
prediction1
# we will use the confusion matrix to #understand our prediction
# Read the documentation for the table() function in RStudio help 
table(irisrand[101:150,5],prediction1)
# you can write the same above line by defining what is the "predicted" 
## table(irisrand[101:150,5],Predicted = prediction1)
plot(classificationmodel1)


