rm(list = ls())
setwd("~/Rudy/Data Science/Kaggle/Titanic")
NNPred<-read.csv("TitanicSolutionNN.csv")
RFPred <-read.csv("TitanicSolutionRF.csv")
SVMPred <-read.csv("TitanicSolutionSVM.csv")

Pred<-data.frame(NNPred[,2],RFPred[,2],SVMPred[,2])
FinalPred <- ifelse(rowSums(Pred) >= 2,1,0)

solution <- read.csv("test.csv",na.strings = c("NA",""))
solution$Survived <- FinalPred
solution<-solution[,c(1,12)]

write.csv(solution,"TitanicFinalPred.csv",row.names = FALSE)


sum(NNPred!=SVMPred);sum(NNPred!=RFPred);sum(RFPred!=SVMPred)
