########################
## SVM Model Parameters:
## MeanAge, MeanAgeOfMiss, MeanAgeOfMs, 
## MeanAgeOfMr, MeanAgeOfMaster, MedianFare
## C,sigma for Gaussian Kernel, d for polynomial kernel
## The ksvm function automatically scales all non-binary features
#######################

rm(list = ls())
setwd("~/Rudy/Data Science/Kaggle/Titanic")
training <- read.csv("train.csv",na.strings = c("NA",""))

library(caret)
library(knitr)
library(kernlab)


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
training$Fare[is.na(training$Fare)] <-MedianFare



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

###################
## MODEL FITTING ##
###################

# # MODEL RBF
# 
# TrainingErrorRBF <- NULL
# CVErrorRBF <- NULL
# 
# for(c in c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10)){
#   for(s in c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10)){
#     svmFit<-ksvm(Survived ~ .,data=training,type="C-svc",kernel ="rbfdot",cross =5, C=c, kpar=list(sigma=s))
#     TrainingErrorRBF<-c(svmFit@error,TrainingErrorRBF)
#     CVErrorRBF<-c(svmFit@cross,CVErrorRBF)
#   }
# }
# 
# TrainingErrorRBF<-matrix(TrainingErrorRBF,ncol=10,byrow = F) ## rows correspond to C, columns to sigma
# CVErrorRBF<-matrix(CVErrorRBF,ncol=10,byrow = F)
# 
# parameterIndexRBF <- which(CVErrorRBF == min(CVErrorRBF), arr.ind = TRUE)
# 
# # returns C=1, sigma=0.05 with a cv error of ~0.17
# # The training error for these parameters is around ~0.16
# # This suggests a high bias, which in turn suggests getting more features
# 
# CVErrorRBF<-CVErrorRBF[parameterIndexRBF[1],parameterIndexRBF[2]]
#
# # Using a polynomial kernel
# 
# TrainingErrorpoly <- NULL
# CVErrorpoly <- NULL
# 
# for(d in 1:8){
#     cat("calculating degree",d,"...","\n")
#     svmFit<-ksvm(Survived ~ .,data=training,type="C-svc",kernel ="polydot",cross =5, kpar=list(degree=d))
#     TrainingErrorpoly<-c(svmFit@error,TrainingErrorpoly)
#     CVErrorpoly<-c(svmFit@cross,CVErrorpoly)
#   
# }
# # plotting some curves
# plot(1:8,TrainingErrorpoly,ylim=c(0,0.3),xlab="degree",ylab="error")
# points(x=1:8,y=CVErrorpoly,col="blue")
# 
# parameterIndexpoly <- which(CVErrorpoly == min(CVErrorpoly))
# CVErrorpolykernel <- CVErrorpoly[parameterIndexpoly]
# TrainingErrorpolykernel <- TrainingErrorpoly[parameterIndexpoly]
# ##also around 17%

################
## PREDICTION ##
################

svmFit<-ksvm(Survived ~ .,data=training[,-1],type="C-svc",kernel ="rbfdot",cross =5, C=1, kpar=list(sigma=0.05))
predSVM<-predict(svmFit,test)

solution <- read.csv("test.csv",na.strings = c("NA",""))
solution$Survived <- predSVM
solution<-solution[,c(1,12)]

write.csv(solution,"TitanicSolutionSVM.csv",row.names = FALSE)

