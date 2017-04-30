library(Matrix)
library(xgboost)
library(corrplot)

auc <- function(actual, predicted)
{
  r <- rank(predicted)
  n_pos <- sum(actual==1)*1.0
  n_neg <- length(actual) - n_pos
  auc <- (sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
  auc
}

#Set the working directory
setwd('C:/Users/DattaDE/Desktop/Machine Learning/Asia Actuarial Analytics Challenge 2017')

#read train and test data files

#Credit default Yes is in train but not in test

ID <- c('CustomerID')
TARGET <- c('Outcome')

train <- read.csv('SAS_Train_Data_v3.csv',stringsAsFactors = F)
test <- read.csv('SAS_Test_Data_v3.csv',stringsAsFactors = F)

# Set up CustomerIDs in submission data 
submissionData <- test[ID]

y_train = train[,"Outcome"]

features<-setdiff(names(train),c("CustomerID","Outcome"))
#features<-setdiff(names(train),c("CustomerID","Outcome","Marital_Status","Education_Level","Housing_Loan","Personal_Loan"))


train<-train[features]
test<-test[features]

ntrain = nrow(train)
train_test = rbind(train, test)

cat_var<-names(train_test)[which(sapply(train_test, is.character))]
num_var<-setdiff(names(train_test),cat_var)

cat_var_len<-c(sapply(test[cat_var],function(x) length(unique(x))))

for(col in cat_var){
  print(col)
  k<-which(names(train_test)==col)
  print(k)
  print(length(unique(train_test[,k])))
  train_test[,k]=as.factor(train_test[,k])
  print(class(train_test[,k]))
}

corrplot(cor(train_test[num_var]))

set.seed(1983)
rand<-sample(c(1:10),ntrain,replace = TRUE)
x_train_all = train_test[1:ntrain,]
x_train = x_train_all[which(rand<=7),]
x_validation = x_train_all[which(rand>7),]
x_test = train_test[(ntrain+1):nrow(train_test),]

#sp<-sparse.model.matrix(~.,data=train_test)

dtrain_all = xgb.DMatrix(sparse.model.matrix(~.-1,data=x_train_all), label=y_train)
dtrain = xgb.DMatrix(sparse.model.matrix(~.-1,data=x_train), label=y_train[which(rand<=7)])
dval = xgb.DMatrix(sparse.model.matrix(~.-1,data=x_validation))
dtest = xgb.DMatrix(sparse.model.matrix(~.-1,data=x_test))

xgb_params = list(
  seed = 1983,
  eta = 0.1,
  objective = 'binary:logistic',
  max_depth = 2L,
  eval_metric="auc",
  colsample_bytree = 0.5,
  subsample = 0.9,
  num_parallel_tree = 5,
  min_child_weight = 10
  #maximize=FALSE
)

res = xgb.cv(xgb_params,
             dtrain_all,
             nrounds=1000,
             nfold=5,
             early_stopping_rounds=20,
             print_every_n = 10,
             verbose= 1)

best_nrounds = res$best_iteration
#best_nrounds = 750

gbdt = xgb.train(xgb_params, dtrain_all,as.integer(best_nrounds/0.8))
imp<-xgb.importance(model = gbdt, feature_names = colnames(dtrain))
xgb.plot.importance(imp)

Pred_xgboos_base = predict(gbdt,dtest)
summary(Pred_xgboos_base)


# Prepare final prediction results 
submissionData$Outcome <- Pred_xgboos_base

# Write prediction results to a file
test.names <- c('CustomerID', 'Outcome')
write.csv(submissionData, file='FinalResults20170428_2.csv',row.names = FALSE)

