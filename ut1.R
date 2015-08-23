library(doParallel)
library(caret)
library(plyr)
library(dplyr)

rm(list=ls())

stopImplicitCluster()
registerDoParallel() #(cores = 7)
getDoParWorkers()

training<-read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',stringsAsFactors = F)
testing<-read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',stringsAsFactors = F)

# Remove rollup summary rows
t.1<-training%>%filter(new_window=='no')
# Remove ID, user, and time variables
t.2<-t.1[8:length(t.1)]
# Remove aggregated values included in the dataset
aggs<-c('max_','min_','avg_','stddev_','var_','kurtosis_','skewness_')
t.2<-t.2[,!grepl(paste('(',aggs,')',collapse = '|',sep=''),names(t.2))]

# Remove variables that contain near 0 variance
nzv<-nearZeroVar(t.2)
t.3<-t.2[,-nzv]
# Remove variables with > 95% null values
t.3<-t.3[,sapply(t.3, function(x) sum(is.na(x)))/nrow(t.3)<=.95]

# Make the response variable a factor
resp<-factor(t.3$classe)

# At this point we should be down to just the sensor values. Impute any remaining missing values to 0
for(i in 1:length(preds)){
  preds[,i]<-ifelse(is.na(preds[,i]),0,preds[,i])
}

# Find and remove predictors that are highly correlated with eachother. 
correlated<-findCorrelation(cor(preds), .9)
preds<-preds[,-correlated]

set.seed(125)

# Split the data into train and test sets with 80% going into the test set
train.idx<-createDataPartition(resp,times = 1,p = .8,list = T)$Resample1
test.idx<-seq_along(resp)[-1*train.idx]

# Save the processed data
save(list=c('preds','resp','testing','training','train.idx','test.idx'),
     file='all.data.RData')

# Setup 5 fold cross validation on the training set
tc<-trainControl(method='cv',allowParallel=T,number=5)

save.image()

# Random forest almost always produces a good model with very little tuning required
fit.rf<-train(x=preds[train.idx,],y=resp[train.idx],method='rf',tuneLength=3)
fit.rf.cm<-confusionMatrix(predict(fit.rf,newdata = preds[test.idx,]),resp[test.idx])
saveRDS(fit.rf,'fit.rf.rds');saveRDS(fit.rf.cm,'fit.rf.cm.rds');

# Boosted trees. Stochatic gradient boosting
fit.gbm<-train(x=preds[train.idx,],y=resp[train.idx],method='gbm',trControl=tc,tuneLength=5)
fit.gbm.cm<-confusionMatrix(predict(fit.gbm,newdata = preds[test.idx,]),resp[test.idx])
saveRDS(fit.gbm,'fit.gbm.rds');saveRDS(fit.gbm.cm,'fit.gbm.cm.rds');

# A variety of linear models provided by lib linear

modelInfo <- list(label = "LiblineaR",
                  library = "LiblineaR",
                  type = c("Classification"),
                  parameters = data.frame(parameter = c('c','t'),
                                          class =     c('numeric','numeric'),
                                          label =     c('c','type')),
                  grid = function(x, y, len = NULL, search = "grid") {
                    library(LiblineaR)
                    C<-heuristicC(as.matrix(x))
                    if(!(is.null(len) | len==1)){
                      C<-seq(C/2,C*2,length.out = len)
                    }
                    data.frame(expand.grid(
                      c=C,
                      t=c(0,1,2,3,4,5,6,7)))
                  },
                  loop = NULL,
                  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
                    library(LiblineaR)
                    
                    m<-LiblineaR(
                      data=as.matrix(x),target=y,
                      type=param[['t']],
                      cost=param[['c']],
                      epsilon=NULL,
                      verbose = F)
                    m
                  },
                  predict = function(modelFit, newdata, submodels = NULL) {
                    unlist(predict(modelFit, as.matrix(newdata)))
                  },
                  prob = NULL,
                  tags = c("Support Vector Machines","Linear Regression", "Linear Classifier"))

fit.ll<-train(x=preds[train.idx,],y=resp[train.idx],method=modelInfo,trControl=tc,preProcess = c('center','scale'),tuneLength = 5)
fit.ll.cm<-confusionMatrix(predict(fit.ll,newdata = preds[test.idx,]),resp[test.idx])
saveRDS(fit.ll,'fit.ll.rds');saveRDS(fit.ll.cm,'fit.ll.cm.rds')