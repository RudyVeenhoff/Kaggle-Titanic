########################
## Neural Networks 
## Model Parameters:
## MeanAge, MeanAgeOfMiss, MeanAgeOfMs,           (we could also try median Age to increase acc)
## MeanAgeOfMr, MeanAgeOfMaster, MedianFare
##
## For scaling: meanAge,sdAge
## 
#######################

rm(list = ls())
setwd("~/Rudy/Data Science/Kaggle/Titanic")
training <- read.csv("train.csv",na.strings = c("NA",""))

library(caret)
library(knitr)
library(neuralnet)
library(beepr)

training$Survived<-as.factor(training$Survived)
Survived <-training[,2] ## reference for later
training$NumberOfRelatives <- training$Parch + training$SibSp
training$Alone <- training$NumberOfRelatives == 0
training$Alone<-as.numeric(training$Alone)
MedianFare <- median(training$Fare,na.rm=T)

# We also make a feature of the title of the person. The title seems to capture the features age and sex, 
# both of which turned out to be important predictors
training$Title <-rep(NA,length(training$PassengerId))
training$Title[grep("[Cc]ol|Dr|[Rr]ev",training$Name)]<-"Mr" #There aren't a lot of these titles in our data
training$Title[grep("[Mm]iss",training$Name)]<-"Miss"
training$Title[grep("[Mm]s|[Dd]ona",training$Name)]<-"Ms"
training$Title[grep("[Mm]rs",training$Name)]<-"Mrs"
training$Title[grep("[Mm]r",training$Name)]<-"Mr"
training$Title[grep("[Mm]aster",training$Name)]<-"Master"
training$Title[is.na(training$Title)] <- "Unknown"

# Number of Parents aboard for children, default set to 1 (assuming that every child was accompanied by at least one adult)
training$Parents<- pmax(1*(training$Title=="Master"),training$Parch *(training$Title=="Master")) + 
  pmax(1*(training$Title=="Miss"),training$Parch *(training$Title=="Miss"))

# Number of children aboard
training$Children<-training$Parch *(training$Title=="Mr") + training$Parch *(training$Title=="Ms")+training$Parch *(training$Title=="Mrs")

# We note that the Age feature contains some NAs. Because Age is used in our decision tree as a splitting variable
# we try to impute the NAs by setting them equal to the mean of the respective title.


## Calculating the mean Age grouped by title, if the title doesnt occur in the dataset we set it equal to mean Age
meanAge <- mean(training$Age,na.rm=T)

if(sum(training$Title == "Miss",na.rm=T) > 0 && !all(is.na(training$Age[training$Title == "Miss"]))){
  MeanAgeOfMiss <- mean((subset(training,training$Title == "Miss"))$Age,na.rm=T)
} else MeanAgeOfMiss <- meanAge

if(sum(training$Title == "Ms",na.rm=T) > 0 && !all(is.na(training$Age[training$Title == "Ms"]))){
  MeanAgeOfMs <- mean((subset(training,training$Title == "Ms"))$Age,na.rm=T)
} else MeanAgeOfMs <- meanAge

if(sum(training$Title == "Mrs",na.rm=T) > 0 && !all(is.na(training$Age[training$Title == "Mrs"]))){
  MeanAgeOfMrs <- mean((subset(training,training$Title == "Mrs"))$Age,na.rm=T)
} else MeanAgeOfMrs <- meanAge 

if(sum(training$Title == "Mr",na.rm=T) > 0 && !all(is.na(training$Age[training$Title == "Mr"]))){
  MeanAgeOfMr <- mean((subset(training,training$Title == "Mr"))$Age,na.rm=T)
} else MeanAgeOfMr <- meanAge

if(sum(training$Title == "Master",na.rm=T) > 0 && !all(is.na(training$Age[training$Title == "Master"]))){
  MeanAgeOfMaster <- mean(subset(training,training$Title == "Master")$Age,na.rm=T)
} else MeanAgeOfMaster <- meanAge



##Imputing the missing age data

for(i in 1:length(training$PassengerId)){
  if(is.na(training$Age[i])) {
    
    if(training$Title[i] == "Master"){
      training$Age[i] <- MeanAgeOfMaster
      
    } else if (training$Title[i] == "Miss"){
      training$Age[i] <- MeanAgeOfMiss
      
    } else if (training$Title[i] == "Mr"){
      training$Age[i] <- MeanAgeOfMr
      
    } else if (training$Title[i] == "Mrs"){
      training$Age[i] <- MeanAgeOfMrs
      
    } else if (training$Title[i] == "Ms"){
      training$Age[i] <- MeanAgeOfMs
      
    }
    
  }
}
training$Title<-as.factor(training$Title)
training$Embarked<-as.character(training$Embarked)
training$Embarked[is.na(training$Embarked)] <-"Unknown"
training$Embarked <- as.factor(training$Embarked)
training$Fare[is.na(training$Fare)] <- MedianFare

## Feature scaling
newMeanAge <- mean(training$Age) ## "new" because it's after imputing
newSdAge <-sd(training$Age)
training$Age<-(training$Age-newMeanAge)/newSdAge

newMeanFare <- mean(training$Fare)
newSdFare <- sd(training$Fare)
training$Fare <- (training$Fare-newMeanFare) / newSdFare



training <-subset(training, select = c("Survived","Pclass","Sex","Age","SibSp",
                                       "Parch","Fare","Embarked","NumberOfRelatives",
                                       "Alone","Title","Parents","Children"))
## PREPARING TEST SET

test <- read.csv("test.csv",na.strings = c("NA",""))
test$NumberOfRelatives <- test$Parch + test$SibSp
test$Alone <- test$NumberOfRelatives == 0
test$Alone<-as.numeric(test$Alone)

# We also make a feature of the title of the person. The title seems to capture the features age and sex, 
# both of which turned out to be important predictors
test$Title <-rep(NA,length(test$PassengerId))
test$Title[grep("Dr",test$Name)]<-"Mr"
test$Title[grep("[Mm]r|[Rr]ev|[Cc]ol",test$Name)]<-"Mr"
test$Title[grep("[Mm]iss",test$Name)]<-"Miss"
test$Title[grep("[Mm]s|[Dd]ona",test$Name)]<-"Ms"
test$Title[grep("[Mm]rs",test$Name)]<-"Mrs"
test$Title[grep("[Mm]aster",test$Name)]<-"Master"
test$Title[is.na(test$Title)] <- "Unknown"



# Number of Parents aboard for children, default set to 1 (assuming that every child was accompanied by at least one adult)
test$Parents<- pmax(1*(test$Title=="Master"),test$Parch *(test$Title=="Master")) + 
  pmax(1*(test$Title=="Miss"),test$Parch *(test$Title=="Miss"))

# Number of children aboard
test$Children<-test$Parch *(test$Title=="Mr") + test$Parch *(test$Title=="Ms")+test$Parch *(test$Title=="Mrs")

# We note that the Age feature contains some NAs. Because Age is used in our decision tree as a splitting variable
# we try to impute the NAs by setting them equal to the mean of the respective title.



##Imputing the missing age data based on training set

for(i in 1:length(test$PassengerId)){
  if(is.na(test$Age[i])) {
    
    if(test$Title[i] == "Master"){
      test$Age[i] <- MeanAgeOfMaster
      
    } else if (test$Title[i] == "Miss"){
      test$Age[i] <- MeanAgeOfMiss
      
    } else if (test$Title[i] == "Mr"){
      test$Age[i] <- MeanAgeOfMr
      
    } else if (test$Title[i] == "Mrs"){
      test$Age[i] <- MeanAgeOfMrs
      
    } else if (test$Title[i] == "Ms"){
      test$Age[i] <- MeanAgeOfMs
      
    } else {test$Age[i] <- meanAge
    
    }
  }
}

# and impute all NAs 
test$Title<-as.factor(test$Title)

test$Embarked<-as.character(test$Embarked)
test$Embarked[is.na(test$Embarked)] <-"Unknown"
test$Embarked <- as.factor(test$Embarked)

test$Fare[is.na(test$Fare)] <- MedianFare

test <-subset(test, select = c("Pclass","Sex","Age","SibSp",
                               "Parch","Fare","Embarked","NumberOfRelatives",
                               "Alone","Title","Parents","Children"))

##make sure labels of factors are the same in training and test.
levels(test$Title)<-union(levels(test$Title),levels(training$Title))
levels(training$Title)<-union(levels(test$Title),levels(training$Title))
levels(test$Embarked)<-union(levels(test$Embarked),levels(training$Embarked))
levels(training$Embarked)<-union(levels(test$Embarked),levels(training$Embarked))

## Feature scaling
test$Age<-(test$Age-newMeanAge)/newSdAge
test$Fare <- (test$Fare-newMeanFare) / newSdFare

training$Pclass<-as.factor(training$Pclass)
training <- model.matrix(~ Pclass + Sex + Age + SibSp + Parch + Fare + 
                           Embarked + NumberOfRelatives + Alone + Title + 
                           Parents + Children,data=training)

Survived <- as.numeric(Survived)-1
training <-cbind(Survived,training)

test$Pclass<-as.factor(test$Pclass)
test <- model.matrix(~ Pclass + Sex + Age + SibSp + Parch + Fare + 
                           Embarked + NumberOfRelatives + Alone + Title + 
                           Parents + Children,data=test)

###################
## MODEL FITTING ##
###################
n <- colnames(training)
f <- as.formula(paste("Survived ~", paste(n[!n %in% c("Survived","(Intercept)")], collapse = " + ")))

# {
# acc <- NULL
# ## Splitting train and test
# for(h in 5:6){
#   set.seed(2*h)
#   inTrain <- createDataPartition(training[,1],p=.7)[[1]]
#   train <- training[inTrain,]
#   cv <- training[-inTrain,]
#   cat("calculating NN with",h," hidden layers...","\n")
#   nnFit <- neuralnet(f, data=train,hidden=c(h),linear.output=F,stepmax=1e+06)
#   predNN<-compute(nnFit,cv[,c(-1,-2)])$net.result
#   pred01NN <- ifelse(predNN > 0.5,1,0)
#   accuracy <- (table(pred01NN,cv[,1])[[1]]+table(pred01NN,cv[,1])[[4]])/length(cv[,1])
#   acc <- c(acc, accuracy)
# }
# beep(3)
# }
################
## PREDICTION ##
################
{
solution <- read.csv("test.csv",na.strings = c("NA",""))

nnFit <- neuralnet(f, data=training,hidden=8,linear.output=F,stepmax=1e+06)
predNN<-compute(nnFit,test[,c(-1)])$net.result
pred01NN <- ifelse(predNN > 0.5,1,0)
solution$Survived <- pred01NN
solution<-solution[,c(1,12)]

write.csv(solution,"TitanicSolutionNN.csv",row.names = FALSE)
}
beep(3)