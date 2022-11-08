library(splines)
library(pROC)
library(dplyr)
library(car)
library(glmnet)
library(MASS)
library(gam)
library(randomForest)
install.packages("gbm")
library(gbm)
install.packages("ALEplot")
library(ALEplot)
library(dplyr)
install.packages("rmarkdown")
library(rmarkdown)

setwd("/Users/olivia/Desktop/IMC/463 ML/Project")
#setwd("/Users/ariadne/Desktop/IMC/Q3/463 Machine Learning I/HW/Group Project")
#data = read.csv("mkt5.csv")
#str(data)

dat = read.csv("mkt5.csv") %>% group_by(SUBSCRIPTIONID) %>% mutate(totsubs=cumsum(NewsLetSubs+NewsLetUnsubs))

#mydat = data.frame(SUBSCRIPTIONID=     c(1,1,1, 2,2,2, 3,3,3,3), newsubs=c(0,1,0, 0,0,0, 1,1,-1,0))
#mydat %>% group_by(SUBSCRIPTIONID) %>% mutate(cumsubs = cumsum(newsubs))
#str(mydat)

#dat = merge(data, mydat, by = "SUBSCRIPTIONID")
str(dat)

train = dat$train==1
table(train)


#1.1    3 variables with no transformation
##logit(3 variables)
fit1 = glm(nextchurn ~ REGULARITY+thismon+t, binomial, dat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot.roc(dat$nextchurn[!train],yhat,print.auc=TRUE)
0.615

hist(dat$REGULARITY)
hist(dat$t)

##stepwise
#fit3 = step(fit)
#mean((dat$nextchurn[!train] - yhat)^2) 
#summary(fit3)


##set up for ridge/lasso
set.seed(12345)
#train1 = ifelse(dat$train==1,TRUE,FALSE)
x = model.matrix(nextchurn ~ REGULARITY+thismon+t, dat)
dim(x)
fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0)
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) # find yhat for best model
yhat1 = predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test set MSE
plot.roc(dat$nextchurn[!train],as.vector(yhat1),print.auc=TRUE)
0.613
length(yhat1)

##lasso 
fit.lasso=glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv=cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat2=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE)
0.613

#mean((dat$nextchurn[!train1] - yhat)^2) # compute test MSE
View(yhat2)
length(yhat2)
length(dat$nextchurn[!train])


#1.2 full model no transformation
##set up for ridge/lasso
set.seed(12345)
#train1 = ifelse(dat$train==1,TRUE,FALSE)
x = model.matrix(nextchurn ~ .-SUBSCRIPTIONID -train, dat)
dim(x)
fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0)
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) # find yhat for best model
yhat1 = predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test set MSE
plot.roc(dat$nextchurn[!train],as.vector(yhat1),print.auc=TRUE)
0.756
length(yhat1)

##lasso 
fit.lasso=glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv=cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat2=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE)
0.758
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test MSE
View(yhat2)
length(yhat2)
length(dat$nextchurn[!train])


#tree
fit2 = tree(nextchurn~.-SUBSCRIPTIONID-train,data=dat[train,],mindev=.0001)
fit2
plot(fit2, type="uniform"); text(fit)
??prune.tree
fit.tree = prune.tree(fit2, best=100)
plot(fit.tree); fit.tree
yhat2=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector") #i have no idea if this is appropriate...
plot.roc(dat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE,col=5,print.auc.y=.2)
0.876



#boosted tree (a little error here; need fixing)
dim(dat[train,])
fit3=gbm(nextchurn~.-SUBSCRIPTIONID-train,data=dat[train,],interaction.depth=2,n.trees=300,shrinkage=.01)
fit3
summary(fit3)
gbm.perf(fit3)
yhat3=predict(fit3,newdata=dat[!train,],n.trees=300,type="response")
hist(yhat3)
length(yhat3)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)
auc=0.853, lift=2.43, 1.72


#2.1 transformation
hist(dat$t)
newdat=dat
newdat$REGULARITY=log(newdat$REGULARITY+1)
newdat$t=log(newdat$t+1)
#newdat$t=newdat$t^2+ ## square
hist(newdat$t)

table(newdat$REGULARITY)
fit.1=glm(nextchurn ~ REGULARITY+thismon+t, binomial, newdat[train,])
yhat.1 = predict(fit.1, newdat[!train,]) 
plot.roc(newdat$nextchurn[!train],yhat.1,print.auc=TRUE)
0.622

##set up for ridge/lasso
set.seed(12345)
#train1 = ifelse(dat$train==1,TRUE,FALSE)
x = model.matrix(nextchurn ~ REGULARITY+thismon+t, newdat)
dim(x)
fit.ridge.1 = glmnet(x[train,], newdat$nextchurn[train], alpha=0)
fit.cv.1 = cv.glmnet(x[train,], newdat$nextchurn[train], alpha=0) # find yhat for best model
yhat.1 = predict(fit.ridge.1, s=fit.cv.1$lambda.min, newx=x[!train,]) 
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test set MSE
plot.roc(newdat$nextchurn[!train],as.vector(yhat.1),print.auc=TRUE)
0.622
length(yhat.1)

##lasso 
fit.lasso.2=glmnet(x[train,], newdat$nextchurn[train], alpha=1) 
fit.cv.2=cv.glmnet(x[train,], newdat$nextchurn[train], alpha=1) 
yhat.2=predict(fit.lasso.2, s=fit.cv.2$lambda.min, newx=x[!train,]) 
plot.roc(newdat$nextchurn[!train],as.vector(yhat.2),print.auc=TRUE)
0.622
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test MSE
View(yhat2)
length(yhat2)
length(dat$nextchurn[!train])


#2.2 transformation full model
##set up for ridge/lasso
set.seed(12345)
#train1 = ifelse(dat$train==1,TRUE,FALSE)
x = model.matrix(nextchurn ~ .-SUBSCRIPTIONID -train, newdat)
dim(x)
fit.ridge.1 = glmnet(x[train,], newdat$nextchurn[train], alpha=0)
fit.cv.1 = cv.glmnet(x[train,], newdat$nextchurn[train], alpha=0) # find yhat for best model
yhat.1 = predict(fit.ridge.1, s=fit.cv.1$lambda.min, newx=x[!train,]) 
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test set MSE
plot.roc(newdat$nextchurn[!train],as.vector(yhat.1),print.auc=TRUE)
0.750
length(yhat.1)

##lasso 
fit.lasso.2=glmnet(x[train,], newdat$nextchurn[train], alpha=1) 
fit.cv.2=cv.glmnet(x[train,], newdat$nextchurn[train], alpha=1) 
yhat.2=predict(fit.lasso.2, s=fit.cv.2$lambda.min, newx=x[!train,]) 
plot.roc(newdat$nextchurn[!train],as.vector(yhat.2),print.auc=TRUE)
0.751
#mean((dat$nextchurn[!train1] - yhat)^2) # compute test MSE
View(yhat2)
length(yhat2)
length(dat$nextchurn[!train])


##roc ignore this part
#testprob = predict(fit, dat[!train,], type="resp")
#hist(testprob)
#plot.roc(dat$nextchurn[train==0], testprob, print.auc=T)

#x = model.matrix(buy ~ ., dat[,-23]) # -23 drops targamnt
#fit.ridge = glmnet(x[train,], dat$buy[train], family="binomial", alpha=0) 
#fit.cv = cv.glmnet(x[train,], dat$buy[train], family="binomial", alpha=0) 
#phat = predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
#roc(dat$buy[!train], phat)

#fit.lasso = glmnet(x[train1,], dat$nextchurn[train1], family="binomial", alpha=1) 
#fit.cv = cv.glmnet(x[train1,], dat$nextchurn[train1], family="binomial", alpha=1) 
#phat = predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train1,]) 
#roc(dat$nextchurn[!train],yhat)

#train2 = (dat$train==1) #create logical train variable
#table(train2)
#testprob = predict(fit.lasso, dat$nextchurn[train1], type = "resp")
#library(pROC)
#plot.roc(mkt5$nextchurn[train2==0], testprob, print.auc = T)

##gam error
library(gam)
install.packages("mgcViz")
library(mgcViz)
fit5 = gam(nextchurn ~ s(REGULARITY)+s(thismon)+s(t), data=dat)
summary(fit5)
par(mfrow=c(1,2))
plot.gam(fit5, se=T, ask=T, scale=2.5)
# mean squared residuals  
mean(fit5$residuals^2)
[1] 0.03331684
# R-squared
1-fit5$deviance/fit5$null.deviance 
[1] 0.02848996

##splines error
library(splines)
#plot(quality$rate, quality$defect, xlab="Rate", ylab="Defects/K", pch=16)
lines(smooth.spline(dat$REGULARITY, dat$nextchurn))

#partial dependence plot error
install.packages("pdp")
library(pdp)
library(ggplot2)
install.packages("randomForest")
library(randomForest)
fit6 = randomForest(nextchurn ~ REGULARITY+thismon+t, data=dat, importance=T)
par(mfrow=c(1,3))
partialPlot(fit6, dat$nextchurn, "REGULARITY") 
partialPlot(fit6, dat$nextchurn, "thismon")
partialPlot(fit6, dat$nextchurn, "t")



##gains 
gains = function(yhat,respond,ngrp=5){
  ans=data.frame(respond=respond,qtile=cut(yhat,breaks=quantile(yhat,probs=seq(0,1,1/ngrp)),
                                           labels=paste("Q",ngrp:1,sep=""),include.lowest=T))
  %>% group_by(qtile) %>% summarise(n=n(),Nrespond=sum(respond),RespRate=Nrespond/n) %>% arrange(desc(qtile)) %>% mutate(CumN=cumsum(n),CumResp=cumsum(Nrespond),CumRespRate=CumResp/CumN) ans %>% mutate(liftResp=CumRespRate/CumRespRate[nrow(ans)])
}


##3 Try Full Model here

str(train)
#Delete the ID variable from the original data
train.1 = train[,-4]
str(train.1)
train.2 = train.1[,-7]
str(train.2)
##logit(full model)
fit2=glm(nextchurn ~., binomial, train.2)
summary(fit2)
#stepwise
fit4 = step(fit2)
yhat = predict(fit4, dat[!train,]) 
mean((dat$nextchurn[!train] - yhat)^2) 
summary(fit4)

##attempt 1
# ridge model with trans
x = model.matrix(nextchurn ~ REGULARITY + sqrt(REGULARITY) +log(REGULARITY)+I(REGULARITY^2)
                 + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE)+ I(NHOMEPAGE^2) + sessions + sqrt(sessions) +log(sessions)+ I(sessions^2)+ thismon  + sqrt(thismon) +log(thismon)+ I(thismon^2)+ t + sqrt(t) +log(t)+ I(t^2)+ PVs + sqrt(PVs) +log(PVs)+ I(PVs^2)+ TopicBreakNews + sqrt(TopicBreakNews) +log(TopicBreakNews)+ I(TopicBreakNews^2)+TopicLocalCom + sqrt(TopicLocalCom) +log(TopicLocalCom)+ I(TopicLocalCom^2)+TopicNatWorld+ sqrt(TopicNatWorld) +log(TopicNatWorld)+ I(TopicNatWorld^2)+TopicLocGov+ sqrt(TopicLocGov) +log(TopicLocGov)+ I(TopicLocGov^2)+TopicStateGov+ sqrt(TopicStateGov) +log(TopicStateGov)+ I(TopicStateGov^2)+TopicNatGov+ sqrt(TopicNatGov) +log(TopicNatGov)+ I(TopicNatGov^2)+TopicHealth+ sqrt(TopicHealth) +log(TopicHealth)+ I(TopicHealth^2)+TopicCrime+ sqrt(TopicCrime) +log(TopicCrime)+ I(TopicCrime^2)+TopicElect+ sqrt(TopicElect) +log(TopicElect)+ I(TopicElect^2)+log(TopicCrime)+ I(TopicCrime^2)+TopicColSport+ sqrt(TopicColSport) +log(TopicColSport )+ I(TopicColSport^2)+TopicProSport+ sqrt(TopicProSport) +log(TopicProSport)+ I(TopicProSport^2)+TopicHSsport+ sqrt(TopicHSsport) +log(TopicHSsport)+ I(TopicHSsport^2)+TopicFireAccident+ sqrt(TopicFireAccident) +log(TopicFireAccident)+ I(TopicFireAccident^2)+TopicImmigration+ sqrt(TopicImmigration) +log(TopicImmigration)+ I(TopicImmigration^2)+TopicEduc+ sqrt(TopicEduc) +log(TopicEduc)+ I(TopicEduc^2)+TopicOpenClose+ sqrt(TopicOpenClose) +log(TopicOpenClose)+ I(TopicOpenClose^2)+TopicJobEcon+ sqrt(TopicJobEcon) +log(TopicJobEcon)+ I(TopicJobEcon^2)+TopicLocBus+ sqrt(TopicLocBus) +log(TopicLocBus)+ I(TopicLocBus^2)+TopicRealEstate+ sqrt(TopicRealEstate) +log(TopicRealEstate)+ I(TopicRealEstate^2)+TopicWeather + sqrt(TopicWeather) +log(TopicWeather)+ I(TopicWeather^2)+TopicEnviron + sqrt(TopicEnviron) +log(TopicEnviron)+ I(TopicEnviron^2)+TopicRestDine + sqrt(TopicRestDine) +log(TopicRestDine)+ I(TopicRestDine^2)+TopicEvent + sqrt(TopicEvent) +log(TopicEvent)+ I(TopicEvent^2)+TopicCelebrity + sqrt(TopicCelebrity) +log(TopicCelebrity)+ I(TopicCelebrity^2)+TopicEntertain + sqrt(TopicEntertain) +log(TopicEntertain)+ I(TopicEntertain^2)+TopicTourism + sqrt(TopicTourism) +log(TopicTourism)+ I(TopicTourism^2)+TopicTraffic + sqrt(TopicTraffic) +log(TopicTraffic)+ I(TopicTraffic^2)+SrcSearch + sqrt(SrcSearch) +log(SrcSearch)+ I(SrcSearch^2)+SrcSocial + sqrt(SrcSocial) +log(SrcSocial)+ I(SrcSocial^2)+MktOutside + sqrt(MktOutside) +log(MktOutside)+ I(MktOutside^2)+DevMobile + sqrt(DevMobile) +log(DevMobile)+ I(DevMobile^2)+DevTablet + sqrt(DevTablet) +log(DevTablet)+ I(DevTablet^2)+DevDesktop + sqrt(DevDesktop) +log(DevDesktop)+ I(DevDesktop^2)+DevApp + sqrt(DevApp) +log(DevApp)+ I(DevApp^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs)+ I(NewsLetUnsubs^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+totsubs,dat)



fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) 
yhat=predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
0.758


#lasso model with trans
fit.lasso = glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
0.762


##attempt 2
x = model.matrix(nextchurn ~ REGULARITY + sqrt(REGULARITY) +log(REGULARITY)+I(REGULARITY^2)
                 + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE)+ I(NHOMEPAGE^2) + sessions + sqrt(sessions) +log(sessions)+ I(sessions^2)+ thismon + sqrt(thismon) +log(thismon)+ I(thismon^2)+ t + sqrt(t) +log(t)+ I(t^2)+ PVs + sqrt(PVs) +log(PVs)+ I(PVs^2)+SrcSearch + sqrt(SrcSearch) +log(SrcSearch)+ I(SrcSearch^2)+SrcSocial + sqrt(SrcSocial) +log(SrcSocial)+ I(SrcSocial^2)+MktOutside + sqrt(MktOutside) +log(MktOutside)+ I(MktOutside^2)+DevMobile + sqrt(DevMobile) +log(DevMobile)+ I(DevMobile^2)+DevTablet + sqrt(DevTablet) +log(DevTablet)+ I(DevTablet^2)+DevDesktop + sqrt(DevDesktop) +log(DevDesktop)+ I(DevDesktop^2)+DevApp + sqrt(DevApp) +log(DevApp)+ I(DevApp^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs)+ I(NewsLetUnsubs^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+totsubs,dat)

fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) 
yhat=predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
0.759


#lasso model with trans
fit.lasso = glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
0.764



##attempt 3
x = model.matrix(nextchurn ~ REGULARITY
                 + NHOMEPAGE + sqrt(NHOMEPAGE) +log(NHOMEPAGE)+ I(NHOMEPAGE^2) + 
                   sessions + sqrt(sessions) +log(sessions)+ I(sessions^2)+ thismon + 
                   sqrt(thismon) +log(thismon)+ I(thismon^2)+ t + sqrt(t) +log(t)+ I(t^2)+ PVs + sqrt(PVs) +log(PVs)+ I(PVs^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs)+ I(NewsLetUnsubs^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+totsubs+sqrt(totsubs) +log(totsubs)+ I(totsubs^2),dat)

fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) 
yhat=predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
0.834


#lasso model with trans
fit.lasso = glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
0.834





















#ignore from here
setwd("/Users/ariadne/Desktop/IMC/Q3/463 Machine Learning I/HW/Group Project")
mkt = read.csv("mkt5.csv")
str(mkt)

library(dplyr)


#Correlation
##– Correlation matrix
#cor(credit[,-c(7:10)])
##– Correlation test
#cor.test(credit$Promotions, credit$Spending)

mkt.train = subset(mkt, train == 1)
str(mkt.train)

train = subset(mkt, train == 1)
test = subset(mkt, train == 0)



#Delete the ID variable from the original data
mkt.train1 = mkt.train[,-4]
str(mkt.train1)
mkt.train2 = mkt.train1[,-7]
str(mkt.train2)

fit1 = glm(nextchurn ~ REGULARITY+thismon+t, binomial, data = mkt.train)
fit1.1 = glm(nextchurn ~ REGULARITY+thismon+t, binomial, data = mkt, subset=(train==1))
summary(fit1)
fit1.2 = lm.ridge(nextchurn ~ REGULARITY+thismon+t, data = mkt, subset=(train==1), lambda=seq(0,0.4,length=41))
summary(fit1.2)

fit2 = glm(nextchurn ~ ., binomial, data = mkt.train2)
summary(fit2)

fit3 = lm(nextchurn ~ .,mkt.train2)
step(fit3)

fit4=glm(nextchurn ~ REGULARITY + thismon + t + TopicOpenClose + 
           SrcSearch + SrcSocial + NewsLetSubs + NewsLetUnsubs, data = mkt.train)
summary(fit4)

install.packages("tree")
library(tree)
fit = tree(nextchurn~., train, mindev=1e-6)

fit2 = prune.tree(fit, best=8) 
par(mfrow=c(1,2))
plot(fit2, type="uniform")
text(fit2, cex=.8)
partition.tree(fit2, cex=.8) 
par(mfrow=c(1,1))
fit2




head(mkt5)
#row.names(college) = college$X
#college$X=NULL
set.seed(12345)
train = runif(nrow(mkt5))<.5    # pick train/test split
dim(mkt5)
table(train) 

#1.1 
##logit(3 variables)
fit1 = glm(nextchurn ~ REGULARITY+thismon+t, binomial, dat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot.roc(dat$nextchurn[!train],yhat,print.auc=TRUE)
0.615






##################################################### #####################
## THIS WORKS!!!!
fit = lm(nextchurn ~ ., dat, subset=train)
plot(fit, which=1, pch=16, cex=.8)
yhat = predict(fit, dat[!train,])
mean((dat$nextchurn[!train] - yhat)^2)       # compute test set MSE
summary(fit)
vif(fit)

# try tree
library(tree)
fit = tree(nextchurn ~ ., dat[train,])
fit
plot(fit, type="uniform")
text(fit, cex=.7)
yhat = predict(fit, newdata=dat[!train,])
mean((dat$nextchurn[!train] - yhat)^2) 
# 0.02949442


# overgrow the tree
fit2 = tree(nextchurn ~ ., dat[train,], mindev= .0001)
fit2
plot(fit2, type="uniform")
text(fit2, cex=.3)
plot(cv.tree(fit2))
yhat = predict(prune.tree(fit2, best=10), newdata=dat[!train,])
mean((dat$nextchurn[!train] - yhat)^2) # 0.02859448
yhat = predict(prune.tree(fit2, best=20), newdata=dat[!train,])
mean((dat$nextchurn[!train] - yhat)^2) # 0.02727203
yhat = predict(prune.tree(fit2, best=30), newdata=dat[!train,])
mean((dat$nextchurn[!train] - yhat)^2) # 0.02672366


### TRYING TOO
library(tree) 
fit = tree(nextchurn ~ ., dat[train,], mindev= .0001)
fit = tree(amus~age+educ, amus, mindev=1e-6)
fit2 = prune.tree(fit, best=8)
par(mfrow=c(1,2))
plot(fit2, type="uniform")
text(fit2, cex=.8)
partition.tree(fit2, cex=.8) ## ERROR Message here
par(mfrow=c(1,1))
fit2


### STILL TRYING
head(dat)
# Random Forest
library(randomForest)
set.seed(12345)
fit  = randomForest(x=dat[train, -2], y=dat$nextchurn[train], xtest=dat[!train,-2], ntree=100)
plot(fit)
varImpPlot(fit)
mean((dat$nextchurn[!train] - fit$test$predicted)^2) # 66.99894 need more trees???

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
