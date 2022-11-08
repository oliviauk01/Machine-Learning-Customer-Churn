library(glmnet); library(car)
library(splines); library(pROC)
library(dplyr); 
library(MASS); library(gam)
library(randomForest)
install.packages("gbm")
library(gbm)
install.packages("ALEplot")
library(ALEplot)
##  predict churn of a media subscription service.

setwd("/Users/olivia/Desktop/IMC/463 ML/Project")
ls()
rm()
getwd()
dat = read.csv("mkt5.csv") %>% group_by(SUBSCRIPTIONID) %>% mutate(totsubs=cumsum(NewsLetSubs+NewsLetUnsubs))
dat = read.csv("mkt5.csv", header = T)
str(mkt5)
train = (mkt5$train==1)
table(train)

topic <- data.frame(mkt5$TopicBreakNews+mkt5$TopicLocalCom+mkt5$TopicNatWorld+mkt5$TopicLocGov+mkt5$TopicStateGov+mkt5$TopicNatGov+
                      mkt5$TopicHealth+mkt5$TopicCrime+mkt5$TopicElect+mkt5$TopicColSport+mkt5$TopicProSport+mkt5$TopicHSsport+mkt5$TopicFireAccident+
                      mkt5$TopicImmigration+mkt5$TopicEduc+mkt5$TopicOpenClose+mkt5$TopicJobEcon+mkt5$TopicLocBus+mkt5$TopicRealEstate+mkt5$TopicWeather+
                      mkt5$TopicEnviron+mkt5$TopicRestDine+mkt5$TopicEvent+mkt5$TopicCelebrity+mkt5$TopicEntertain+mkt5$TopicTourism+mkt5$TopicTraffic)


### ARIADNE'S CODE
#Final Project
## create new variables
news <- data.frame(mkt5$TopicBreakNews, mkt5$TopicLocalCom, mkt5$TopicNatWorld)
govt <- data.frame(mkt5$TopicLocGov+mkt5$TopicStateGov+mkt5$TopicNatGov)
sport <- data.frame(mkt5$TopicColSport+mkt5$TopicProSport+mkt5$TopicHSsport)
econ <- data.frame(mkt5$TopicOpenClose+mkt5$TopicJobEcon+mkt5$TopicLocBus+mkt5$TopicRealEstate)
envr <- data.frame(mkt5$TopicHealth+mkt5$TopicWeather+mkt5$TopicEnviron)
ent <- data.frame(mkt5$TopicRestDine+mkt5$TopicEvent+mkt5$TopicCelebrity+mkt5$TopicEntertain+mkt5$TopicTourism+mkt5$TopicTraffic)

social <- data.frame(mkt5$TopicCrime+mkt5$TopicElect+mkt5$TopicFireAccident+mkt5$TopicImmigration+mkt5$TopicEduc)




##NOT SURE
mkt5$news <- data.frame(mkt5$TopicBreakNews, mkt5$TopicLocalCom, mkt5$TopicNatWorld)
mkt5$govt <- data.frame(mkt5$TopicLocGov+mkt5$TopicStateGov+mkt5$TopicNatGov)
mkt5$sport <- data.frame(mkt5$TopicColSport+mkt5$TopicProSport+mkt5$TopicHSsport)
mkt5$econ <- data.frame(mkt5$TopicOpenClose+mkt5$TopicJobEcon+mkt5$TopicLocBus+mkt5$TopicRealEstate)
mkt5$envr <- data.frame(mkt5$TopicHealth+mkt5$TopicWeather+mkt5$TopicEnviron)
mkt5$ent <- data.frame(mkt5$TopicRestDine+mkt5$TopicEvent+mkt5$TopicCelebrity+mkt5$TopicEntertain+mkt5$TopicTourism+mkt5$TopicTraffic)
mkt5$social <- data.frame(mkt5$TopicCrime+mkt5$TopicElect+mkt5$TopicFireAccident+mkt5$TopicImmigration+mkt5$TopicEduc)

str(news)
str(mkt5$TopicBreakNews)
mkt5$t = as.factor(mkt5$t)
mkt5$t = as.numeric(mkt5$t)

## DESCRIPTIVE
table(mkt5$PVs)
hist(mkt5$PVs)
table(mkt5$train)
table(mkt5$thismon)

## customer time (month number of the customer’s life)
table(mkt5$t)
hist(mkt5$t)
table(mkt5$t)

## REGULARITY - the number of days
table(mkt5$REGULARITY)
hist(mkt5$REGULARITY)

##sessions - the number of sessions
table(mkt5$sessions)
hist(mkt5$sessions)

table(mkt5$nextchurn)

##
fit1 = glm(nextchurn ~ REGULARITY+sessions+NHOMEPAGE+thismon+t+PVs+
             TopicBreakNews+TopicLocalCom+TopicNatWorld+TopicLocGov+TopicStateGov+TopicNatGov+
             TopicHealth+TopicCrime+TopicElect+TopicColSport+TopicProSport+TopicHSsport+TopicFireAccident+
             TopicImmigration+TopicEduc+TopicOpenClose+TopicJobEcon+TopicLocBus+TopicRealEstate+TopicWeather+
             TopicEnviron+TopicRestDine+TopicEvent+TopicCelebrity+TopicEntertain+TopicTourism+TopicTraffic+
             SrcSearch+SrcSocial+MktOutside+DevMobile+DevTablet+DevDesktop+DevApp+NewsLetSubs+NewsLetUnsubs,
           family="binomial"(link=logit), data = mkt5, subset=(train==1))
summary(fit1)
vif(fit1)
str(mkt5$TopicOpenClose)
fit2 = glm(nextchurn ~ REGULARITY+sessions+NHOMEPAGE+thismon+t+PVs+
             TopicOpenClose+TopicLocBus+TopicTraffic+
             SrcSearch+SrcSocial+MktOutside+DevMobile+DevTablet+DevDesktop+DevApp+NewsLetSubs+NewsLetUnsubs,
           family="binomial"(link=logit), data = mkt5, subset=(train==1))
summary(fit2)
vif(fit2)


##??
fit3 = glm(nextchurn ~ REGULARITY+sessions+NHOMEPAGE+thismon+t+PVs+
             news+govt+ SrcSearch+SrcSocial+MktOutside+DevMobile+DevTablet+DevDesktop+DevApp+NewsLetSubs+NewsLetUnsubs,
           family="binomial"(link=logit), data = mkt5, subset=(train==1))
summary(fit3)
testprob = predict(fit2, mkt5[!train,], type="resp")
hist(testprob)

## =====TRANSFORM VARIABLES======
fit4 = glm(nextchurn ~ REGULARITY+sessions+NHOMEPAGE+thismon+t+PVs+
             mkt5$news+ mkt5$govt + mkt5$sport + mkt5$econ + mkt5$envr + mkt5$ent + mkt5$social + 
  SrcSearch+SrcSocial+MktOutside+DevMobile+DevTablet+DevDesktop+DevApp+NewsLetSubs+NewsLetUnsubs, 
family="binomial"(link=logit), data = mkt5, subset=(train==1))


plot(mkt5)
cor(mkt5)
de <- data.frame(mkt5$t,mkt5$REGULARITY,mkt5$NHOMEPAGE,mkt5$sessions,mkt5$nextchurn,mkt5$thismon,mkt5$PVs)
cor(de)
plot(de)
vif(fit1)


fit = glm(targbuy ~ r + fitem + m, family = "binomial", data=all, subset=(train==1))
summary(fit)





## TREE

head(mkt5)
#row.names(college) = college$X
#college$X=NULL
set.seed(12345)
train = runif(nrow(mkt5))<.5    # pick train/test split
dim(mkt5)
table(train) 


# try tree
head(mkt5)
str(mkt5)
library(tree)
fit = tree(nextchurn ~ ., mkt5[train,])
fit
plot(fit, type="uniform")
text(fit, cex=.7)


yhat = predict(fit, newdata=college[!train,])
mean((college$Apps[!train] - yhat)^2) # 109.7879

## fit2
fit2 = tree(nextchurn ~ REGULARITY+sessions+NHOMEPAGE+thismon+t+PVs+
              SrcSearch+SrcSocial+MktOutside+DevMobile+DevTablet+DevDesktop+DevApp+NewsLetSubs+NewsLetUnsubs, mkt5[train,])

fit2
plot(fit2, type="uniform")
text(fit2, cex=.7)
yhat = predict(fit2, newdata=college[!train,])
mean((mkt5$nextchurn[!train] - yhat)^2) # 109.7879

##### TEST
fit2 = tree(nextchurn ~ REGULARITY+sessions+NHOMEPAGE+thismon+t+PVs+
              SrcSearch+SrcSocial+MktOutside+DevMobile+DevTablet+DevDesktop+DevApp+NewsLetSubs+NewsLetUnsubs, mkt5)

fit2
## LOG TRANSFORMATION
## fit3
fit3 = tree(nextchurn ~ log(REGULARITY) + log(sessions) + log(NHOMEPAGE)+thismon+t+log(PVs)+
              log(SrcSearch)+log(SrcSocial) + log(MktOutside) + log(DevMobile) + log(DevTablet) + 
              log(DevDesktop)+ log(DevApp)+ log(NewsLetSubs)+ log(NewsLetUnsubs), mkt5[train,])

fit3
fit4 = prune.tree(fit3, best=8)
plot(fit, type="uniform")
text(fit, cex=.3)
plot(cv.tree(fit))

hist(mkt5$SrcSocial)

hist(mkt5$MktOutside)
hist(mkt5$DevMobile)
hist(mkt5$DevMobile)
hist(mkt5$DevTablet)
hist(mkt5$DevDesktop)
hist(mkt5$DevApp)
hist(mkt5$NewsLetSubs)
hist(mkt5$NewsLetUnsubs)




str(mkt5$t)
# overgrow the tree
fit = tree(nextchurn ~ ., mkt5[train,], mindev= .0001)
fit
plot(fit, type="uniform")
text(fit, cex=.3)
plot(cv.tree(fit))
yhat = predict(prune.tree(fit, best=10), newdata=college[!train,])
mean((college$Apps[!train] - yhat)^2) # 95.93813
yhat = predict(prune.tree(fit, best=20), newdata=college[!train,])
mean((college$Apps[!train] - yhat)^2) # 89.3257
yhat = predict(prune.tree(fit, best=30), newdata=college[!train,])
mean((college$Apps[!train] - yhat)^2) # 92.32201

# Random Forest
library(randomForest)
set.seed(12345)
fit  = randomForest(x=college[train, -2], y=college$Apps[train], xtest=college[!train,-2], ntree=100)
plot(fit)
varImpPlot(fit)
mean((college$Apps[!train] - fit$test$predicted)^2) # 66.99894 need more trees???

# Random forest with more trees
set.seed(12345)
fit  = randomForest(x=college[train, -2], y=college$Apps[train], xtest=college[!train,-2], ntree=1000)
plot(fit)
mean((college$Apps[!train] - fit$test$predicted)^2) # 65.02892

# now try bagging
set.seed(12345)
fit  = randomForest(x=college[train, -2], y=college$Apps[train], xtest=college[!train,-2], ntree=100, mtry=17)
mean((college$Apps[!train] - fit$test$predicted)^2) #  53.60469 need more trees???

# bagging with more trees
set.seed(12345)
fit  = randomForest(x=college[train, -2], y=college$Apps[train], xtest=college[!train,-2], ntree=500, mtry=17)
plot(fit)
mean((college$Apps[!train] - fit$test$predicted)^2) #  53.10197

# bagging with even more trees
set.seed(12345)
fit  = randomForest(x=college[train, -2], y=college$Apps[train], xtest=college[!train,-2], ntree=1000, mtry=17)
mean((college$Apps[!train] - fit$test$predicted)^2) #  53.32566

# boosted tree
library(gbm)
set.seed(12345)
fit = gbm(Apps ~ ., data=college[train,], interaction.depth=2, cv.folds=10, n.trees=5000)
gbm.perf(fit)
summary(fit)
yhat = predict(fit, newdata=college[!train,], n.trees=500)
mean((college$Apps[!train] - yhat)^2) # 58.5233 Note: try more trees

# boosted tree with more trees
yhat = predict(fit, newdata=college[!train,], n.trees=500)
mean((college$Apps[!train] - yhat)^2) # 70.51755 Note: try more trees

# boosted tree with even more trees
yhat = predict(fit, newdata=college[!train,], n.trees=10000)
mean((college$Apps[!train] - yhat)^2) # 58.41246 Note: try more trees

# boosted tree with learning rate .01
set.seed(12345)
fit = gbm(Apps ~ ., data=college[train,], interaction.depth=2, n.trees=10000, shrinkage=.01, cv.folds=10)
gbm.perf(fit)
yhat = predict(fit, newdata=college[!train,], n.trees=1000)
mean((college$Apps[!train] - yhat)^2) # 63.54462 
yhat = predict(fit, newdata=college[!train,], n.trees=2000)
mean((college$Apps[!train] - yhat)^2) # 58.93312 
yhat = predict(fit, newdata=college[!train,], n.trees=5000)
mean((college$Apps[!train] - yhat)^2) # 

# boosted tree with learning rate .005
set.seed(12345)
fit = gbm(Apps ~ ., data=college[train,], interaction.depth=2, n.trees=10000, shrinkage=.005, cv.folds=10)
gbm.perf(fit)
yhat = predict(fit, newdata=college[!train,], n.trees=3000)
mean((college$Apps[!train] - yhat)^2) # 60.53273 
yhat = predict(fit, newdata=college[!train,], n.trees=5000)
mean((college$Apps[!train] - yhat)^2) # 58.05411 

# boosted tree with increased learning rate and more trees
yhat = predict(fit, newdata=college[!train,], n.trees=10000)
mean((college$Apps[!train] - yhat)^2) # 56.09985 
yhat = predict(fit, newdata=college[!train,], n.trees=20000)
mean((college$Apps[!train] - yhat)^2) # 56.09985 

# boosted tree with increased learning rate .01, lower interaction depth
fit = gbm(Apps ~ ., data=college[train,], interaction.depth=1, n.trees=20000, shrinkage=.01)
yhat = predict(fit, newdata=college[!train,], n.trees=500)
mean((college$Apps[!train] - yhat)^2) # 56.76147 
yhat = predict(fit, newdata=college[!train,], n.trees=1000)
mean((college$Apps[!train] - yhat)^2) # 56.76147 
yhat = predict(fit, newdata=college[!train,], n.trees=5000)
mean((college$Apps[!train] - yhat)^2) # 56.76147 
yhat = predict(fit, newdata=college[!train,], n.trees=10000)
mean((college$Apps[!train] - yhat)^2) # 56.76147 
yhat = predict(fit, newdata=college[!train,], n.trees=20000)
mean((college$Apps[!train] - yhat)^2) # 56.76147 

# clean up
rm(college, train, fit, yhat, fit2, fit.lasso, fit.ridge, fit.cv)

