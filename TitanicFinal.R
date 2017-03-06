
rm(list = ls())
setwd("~/Rudy/Data Science/Kaggle/Titanic")
training <- read.csv("train.csv",na.strings = c("NA",""))
test <- read.csv("test.csv",na.strings = c("NA",""))
TrainTest <- rbind(training[,-2],test)
Survived <-as.factor(training[,2])

library(caret)
library(knitr)
library(randomForest)
library(kernlab)
library(neuralnet)
library(beepr)

TrainTest$NumberOfRelatives <- TrainTest$Parch + TrainTest$SibSp
TrainTest$Alone <- TrainTest$NumberOfRelatives == 0
TrainTest$Alone<-as.numeric(TrainTest$Alone)
MedianFare <- median(TrainTest$Fare,na.rm=T)

# We also make a feature of the title of the person. The title seems to capture the features age and sex, 
# both of which turned out to be important predictors
TrainTest$Title <-rep(NA,length(TrainTest$PassengerId))
TrainTest$Title[grep("[Cc]ol|Dr|[Rr]ev",TrainTest$Name)]<-"Mr" #There aren't a lot of these titles in our data
TrainTest$Title[grep("[Mm]iss",TrainTest$Name)]<-"Miss"
TrainTest$Title[grep("[Mm]s|[Dd]ona",TrainTest$Name)]<-"Ms"
TrainTest$Title[grep("[Mm]rs",TrainTest$Name)]<-"Mrs"
TrainTest$Title[grep("[Mm]r",TrainTest$Name)]<-"Mr"
TrainTest$Title[grep("[Mm]aster",TrainTest$Name)]<-"Master"
TrainTest$Title[is.na(TrainTest$Title)] <- "Unknown"

# Number of Parents aboard for children, default set to 1 (assuming that every child was accompanied by at least one adult)
TrainTest$Parents<- pmax(1*(TrainTest$Title=="Master"),TrainTest$Parch *(TrainTest$Title=="Master")) + 
  pmax(1*(TrainTest$Title=="Miss"),TrainTest$Parch *(TrainTest$Title=="Miss"))

# Number of children aboard
TrainTest$Children<-TrainTest$Parch *(TrainTest$Title=="Mr") + TrainTest$Parch *(TrainTest$Title=="Ms")+TrainTest$Parch *(TrainTest$Title=="Mrs")

# We note that the Age feature contains some NAs. Because Age is used in our decision tree as a splitting variable
# we try to impute the NAs by setting them equal to the mean of the respective title.


## Calculating the mean Age grouped by title, if the title doesnt occur in the dataset we set it equal to mean Age
meanAge <- mean(TrainTest$Age,na.rm=T)

if(sum(TrainTest$Title == "Miss",na.rm=T) > 0 && !all(is.na(TrainTest$Age[TrainTest$Title == "Miss"]))){
  MeanAgeOfMiss <- mean((subset(TrainTest,TrainTest$Title == "Miss"))$Age,na.rm=T)
} else MeanAgeOfMiss <- meanAge

if(sum(TrainTest$Title == "Ms",na.rm=T) > 0 && !all(is.na(TrainTest$Age[TrainTest$Title == "Ms"]))){
  MeanAgeOfMs <- mean((subset(TrainTest,TrainTest$Title == "Ms"))$Age,na.rm=T)
} else MeanAgeOfMs <- meanAge

if(sum(TrainTest$Title == "Mrs",na.rm=T) > 0 && !all(is.na(TrainTest$Age[TrainTest$Title == "Mrs"]))){
  MeanAgeOfMrs <- mean((subset(TrainTest,TrainTest$Title == "Mrs"))$Age,na.rm=T)
} else MeanAgeOfMrs <- meanAge 

if(sum(TrainTest$Title == "Mr",na.rm=T) > 0 && !all(is.na(TrainTest$Age[TrainTest$Title == "Mr"]))){
  MeanAgeOfMr <- mean((subset(TrainTest,TrainTest$Title == "Mr"))$Age,na.rm=T)
} else MeanAgeOfMr <- meanAge

if(sum(TrainTest$Title == "Master",na.rm=T) > 0 && !all(is.na(TrainTest$Age[TrainTest$Title == "Master"]))){
  MeanAgeOfMaster <- mean(subset(TrainTest,TrainTest$Title == "Master")$Age,na.rm=T)
} else MeanAgeOfMaster <- meanAge



##Imputing the missing age data

for(i in 1:length(TrainTest$PassengerId)){
  if(is.na(TrainTest$Age[i])) {
    
    if(TrainTest$Title[i] == "Master"){
      TrainTest$Age[i] <- MeanAgeOfMaster
      
    } else if (TrainTest$Title[i] == "Miss"){
      TrainTest$Age[i] <- MeanAgeOfMiss
      
    } else if (TrainTest$Title[i] == "Mr"){
      TrainTest$Age[i] <- MeanAgeOfMr
      
    } else if (TrainTest$Title[i] == "Mrs"){
      TrainTest$Age[i] <- MeanAgeOfMrs
      
    } else if (TrainTest$Title[i] == "Ms"){
      TrainTest$Age[i] <- MeanAgeOfMs
      
    }
    
  }
}
TrainTest$Title<-as.factor(TrainTest$Title)
TrainTest$Embarked<-as.character(TrainTest$Embarked)
TrainTest$Embarked[is.na(TrainTest$Embarked)] <-"Unknown"
TrainTest$Embarked <- as.factor(TrainTest$Embarked)
TrainTest$Fare[is.na(TrainTest$Fare)] <- MedianFare

## Adressing the Fares
TrainTest$Name<-as.character(TrainTest$Name)
TrainTest$FamilyName<-as.character(lapply(strsplit(TrainTest$Name,split=",") , function(l) l[[1]][1]))
uniqueNames <- unique(TrainTest$FamilyName)

for(k in 1:length(uniqueNames)){
  if(sum(uniqueNames[k] == TrainTest$FamilyName) > 1 ){
    family <- subset(TrainTest,TrainTest$FamilyName == uniqueNames[k])
    if(length(unique(family$Fare))==1){
      fare <-  family$Fare[1]/length(family$FamilyName)
      TrainTest$Fare[TrainTest$FamilyName==uniqueNames[k]]<- fare
    }
  }
}



## Are people travelling in groups (i.e. having the same ticket number)
uniqueTickets <- unique(TrainTest$Ticket)
TrainTest$GroupSize <- 1
for(k in 1:length(uniqueTickets)){
  if(sum(uniqueTickets[k] == TrainTest$Ticket) > 1 ){
      GroupSize <- sum(uniqueTickets[k] == TrainTest$Ticket)
      TrainTest$GroupSize[uniqueTickets[k] == TrainTest$Ticket]<- GroupSize
  }
}

TrainTest <-subset(TrainTest, select =c("Pclass","Sex","Age","SibSp",
                                      "Parch","Fare","Embarked","Title","NumberOfRelatives",
                                      "Parents","Children","GroupSize"))
train<-TrainTest[1:891,]
test<-TrainTest[892:1309,]

#######################
## MODEL FITTING  RF ##
#######################

rfFit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch +
                        Fare + Embarked + Title +NumberOfRelatives +GroupSize
                      ,
                      data=train, ntree=100)


plot(rfFit,main="OOB estimate of error")
rfFit$err.rate[100,]
# Around 17%
varImpPlot(rfFit)

#######################
## MODEL FITTING SVM ##
#######################


TrainingErrorRBF <- NULL
CVErrorRBF <- NULL

for(c in c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10)){
  for(s in c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10)){
    svmFit<-ksvm(Survived ~ .,data=train,type="C-svc",kernel ="rbfdot",cross =5, C=c, kpar=list(sigma=s))
    TrainingErrorRBF<-c(svmFit@error,TrainingErrorRBF)
    CVErrorRBF<-c(svmFit@cross,CVErrorRBF)
  }
}

TrainingErrorRBF<-matrix(TrainingErrorRBF,ncol=10,byrow = F) ## rows correspond to C, columns to sigma
CVErrorRBF<-matrix(CVErrorRBF,ncol=10,byrow = F)

parameterIndexRBF <- which(CVErrorRBF == min(CVErrorRBF), arr.ind = TRUE)

c <-c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10)[parameterIndexRBF[1]]
sigma <-c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10)[parameterIndexRBF[2]]
svmFit<-ksvm(Survived ~ .,data=train[,-1],type="C-svc",kernel ="rbfdot",cross =5, C=c, kpar=list(sigma=sigma))
predSVM<-predict(svmFit,test)

#######################
## MODEL FITTING NN  ##
#######################

TrainTest$Pclass<-as.factor(TrainTest$Pclass)
TrainTestNN <- model.matrix(~ Pclass + Sex + Age + SibSp + Parch +
                           Fare + Embarked + Title + NumberOfRelatives + GroupSize,
                          data=TrainTest)


trainNN <- TrainTestNN[1:891,]
testNN <- TrainTestNN[892:1309,]

Survived <- as.numeric(Survived)-1

n <- colnames(trainNN)
f <- as.formula(paste("Survived ~", paste(n[!n %in% c("Survived","(Intercept)")], collapse = " + ")))

nnFit <- neuralnet(f, data=trainNN,hidden=c(8),linear.output=F,stepmax=1e+07)

################
## PREDICTION ##
################

predRF <- predict(rfFit, newdata=test,type="class")
predSVM <- predict(svmFit,test)
predNN<-compute(nnFit,testNN[,c(-1)])$net.result
predNN <- ifelse(predNN > 0.5,1,0)

pred<-data.frame(predNN,as.numeric(predRF)-1,as.numeric(predSVM)-1)

FinalPred <- ifelse(rowSums(pred) >= 2,1,0)
solution <- read.csv("test.csv",na.strings = c("NA",""))
solution$Survived <- FinalPred
solution<-solution[,c(1,12)]
write.csv(solution,"TitanicSolutionFinal1.csv",row.names = FALSE)
