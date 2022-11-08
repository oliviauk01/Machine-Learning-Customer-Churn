---
  title: "Appendix"
output:
  html_document:
  df_print: paged
---
  
  #Setup
setwd("/Users/olivia/Desktop/IMC/463 ML/Project")
getwd()
library(splines)
library(pROC)
library(car)
library(glmnet)
library(MASS)
library(gam)
library(randomForest)
install.packages("gbm")
library(gbm)
library(dplyr)
install.packages("tree")
library(tree)
```

#Transformation (total subscription)

dat = read.csv("mkt5.csv") %>% group_by(SUBSCRIPTIONID) %>% mutate(totsubs=cumsum(NewsLetSubs+NewsLetUnsubs))
rdat = read.csv("mkt5.csv") %>% group_by(SUBSCRIPTIONID) %>% arrange(thismon) %>% mutate(totsubs=cumsum(NewsLetSubs+NewsLetUnsubs))

str(dat)
hist(dat$thismon)
hist(dat$nextchurn)
cor(dat$thismon,dat$nextchurn)
table(dat$thismon,dat$nextchurn)
plot(dat$thismon,dat$nextchurn)


#Train
train = dat$train==1
table(train)

rtrain = rdat$train==1
table(rtrain)

#1 Regularity Model without Transformation
##1.1 Logistic Regression
fit1 = glm(nextchurn ~ REGULARITY+thismon+t, binomial, dat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot <- plot.roc(dat$nextchurn[!train],yhat,col=1,legacy.axes=T,print.auc=TRUE,print.auc.col=1)

##1.2 Ridge
```{r}
set.seed(12345)
x = model.matrix(nextchurn ~ REGULARITY+thismon+t, dat)
dim(x)
fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0)
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) # find yhat for best model
yhat1 = predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat1),add=T,col=2,print.auc=TRUE, print.auc.y=.6,print.auc.col=2)

###
fit = glm(default ~ log(downpmt+1)+pmttype+use+age+gender, binomial, default, subset=ok)
plot.roc(default$default[ok], fit$fitted.values, legacy.axes=T,
         print.auc=T, print.auc.x=1, print.auc.y=.9)
fit2 = glm(default ~ pmttype+age+gender, binomial, default, subset=ok)
plot.roc(default$default[ok], fit2$fitted.values, add=T, col=2,
         print.auc=T, print.auc.x=.7, print.auc.y=.6, print.auc.col=2)
###
##1.3 Lasso
fit.lasso=glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv=cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat2=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat2),add=T,col=3,print.auc=TRUE, print.auc.x=.3, print.auc.y=.8,print.auc.col=3)



##1.4 Tree
fit2 = tree(nextchurn ~ REGULARITY+thismon+t,data=dat[train,],mindev=.0001)
fit2
#plot(fit2, type="uniform")
fit.tree = prune.tree(fit2, best=100)
#plot(fit.tree); fit.tree
yhat5=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector")
plot.roc(dat$nextchurn[!train],as.vector(yhat5),add=T,col=4,print.auc=TRUE, print.auc.x=.6,print.auc.y=.7,print.auc.col=4)


##1.5 RF
fit4 = randomForest(nextchurn ~ REGULARITY+thismon+t,data=dat[train,], n.tree = 300)
summary(fit4)
yhat4=predict(fit4,newdata=dat[!train,],n.trees=300,type="response")
plot.roc(dat$nextchurn[!train],as.vector(yhat4),legacy.axes=T,add=T,print.auc=TRUE,col=5,print.auc.x=.8,print.auc.y=.6,print.auc.col=5)


##1.6 GBM
fit3=gbm(nextchurn ~ REGULARITY+thismon+t,data=dat[train,],interaction.depth=2,n.trees=3000,shrinkage=.01)
summary(fit3)
gbm.perf(fit3)
yhat3=predict(fit3,newdata=dat[!train,],n.trees=300,type="response")
length(yhat3)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,add=T,col=6,print.auc.x=.8, print.auc.y=.4,print.auc.col=6)

## COMBINE AUC
plot <- plot.roc(dat$nextchurn[!train],yhat,col=1,legacy.axes=T,print.auc=TRUE,print.auc.col=1) #0.615
plot.roc(dat$nextchurn[!train],as.vector(yhat1),add=T,col=2,print.auc=TRUE, print.auc.x=.7,print.auc.y=.4,print.auc.col=2) #0.613
plot.roc(dat$nextchurn[!train],as.vector(yhat2),add=T,col=3,print.auc=TRUE, print.auc.x=.8, print.auc.y=.3,print.auc.col=3) #0.613
plot.roc(dat$nextchurn[!train],as.vector(yhat5),add=T,col=4,print.auc=TRUE, print.auc.x=.6,print.auc.y=.7,print.auc.col=4) #0.742
plot.roc(dat$nextchurn[!train],as.vector(yhat4),legacy.axes=T,add=T,print.auc=TRUE,col=5,print.auc.x=.8,print.auc.y=.6,print.auc.col=5) #0.742
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,add=T,col=6,print.auc.x=.9, print.auc.y=.35,print.auc.col=6) #0.713
##1.7 GAM
```{r}
fit4 = gam(nextchurn~s(REGULARITY)+s(thismon)+s(t), data = dat, subset = (train==1))
summary(fit4)




#2 Regularity Model with Transformation
##2.1 Logistic Regression
newdat=dat
newdat$REGULARITY=log(newdat$REGULARITY+1)
newdat$t=log(newdat$t+1)
table(newdat$REGULARITY)
fit.1=glm(nextchurn ~ REGULARITY+thismon+t, binomial, newdat[train,])
yhat.1 = predict(fit.1, newdat[!train,]) 
plot.roc(newdat$nextchurn[!train],yhat.1,print.auc=TRUE)
```

##2.2 Ridge
set.seed(12345)
x = model.matrix(nextchurn ~ REGULARITY+thismon+t, newdat)
dim(x)
fit.ridge.1 = glmnet(x[train,], newdat$nextchurn[train], alpha=0)
fit.cv.1 = cv.glmnet(x[train,], newdat$nextchurn[train], alpha=0) 
yhat.1 = predict(fit.ridge.1, s=fit.cv.1$lambda.min, newx=x[!train,]) 
plot.roc(newdat$nextchurn[!train],as.vector(yhat.1),print.auc=TRUE)
```

##2.3 Lasso
fit.lasso.2=glmnet(x[train,], newdat$nextchurn[train], alpha=1) 
fit.cv.2=cv.glmnet(x[train,], newdat$nextchurn[train], alpha=1) 
yhat.2=predict(fit.lasso.2, s=fit.cv.2$lambda.min, newx=x[!train,]) 
plot.roc(newdat$nextchurn[!train],as.vector(yhat.2),print.auc=TRUE)
```

##2.4 Tree
fit2 = tree(nextchurn ~ REGULARITY+thismon+t,data=newdat[train,],mindev=.0001)
fit2
plot(fit2, type="uniform")
fit.tree = prune.tree(fit2, best=100)
plot(fit.tree); fit.tree
yhat2=predict(fit.tree,newdata=newdat[!train,],n.trees=300,type="vector")
plot.roc(newdat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE,col=5,print.auc.y=.2)
```

##2.5 RF
fit4 = randomForest(nextchurn ~ REGULARITY+thismon+t,data=newdat[train,], n.tree = 300)
summary(fit4)
yhat4=predict(fit4,newdata=newdat[!train,],n.trees=300,type="response")
plot.roc(newdat$nextchurn[!train],as.vector(yhat4),print.auc=TRUE,col=5,print.auc.y=.2)
```

##2.6 GBM
fit3=gbm(nextchurn ~ REGULARITY+thismon+t,data=newdat[train,],interaction.depth=2,n.trees=3000,shrinkage=.01)
summary(fit3)
gbm.perf(fit3)
yhat3=predict(fit3,newdata=newdat[!train,],n.trees=300,type="response")
length(yhat3)
length(dat$nextchurn[!train])
plot.roc(newdat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)
```

##2.7 GAM
```{r}
fit4 = gam(nextchurn~s(REGULARITY)+s(thismon)+s(t), data = newdat, subset = (train==1))
summary(fit4)
```


#3 Full Model without Transformation
##3.1 Logistic Regression
```{r}
fit1 = glm(nextchurn ~.-SUBSCRIPTIONID-train, binomial, dat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot.roc(dat$nextchurn[!train],yhat,print.auc=TRUE)

fit1 = glm(nextchurn ~.-SUBSCRIPTIONID-train, binomial, rdat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot.roc(dat$nextchurn[!train],yhat,print.auc=TRUE)
```

##3.2 Ridge
```{r}
set.seed(12345)
x = model.matrix(nextchurn ~ .-SUBSCRIPTIONID -train, dat)
dim(x)
fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0)
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0)
yhat1 = predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat1),print.auc=TRUE)
```

##3.3 Lasso
```{r}
fit.lasso=glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv=cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat2=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE)
```

##3.4 Tree
```{r}
fit2 = tree(nextchurn~.-SUBSCRIPTIONID-train,data=dat[train,],mindev=.0001)
fit2
plot(fit2, type="uniform")
fit.tree = prune.tree(fit2, best=100)
plot(fit.tree); fit.tree
yhat2=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector")
plot.roc(dat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE,col=5,print.auc.y=.2)
```

##3.5 RF
```{r}
fit4 = randomForest(nextchurn~.-SUBSCRIPTIONID-train,data=dat[train,], n.tree = 300)
summary(fit4)
yhat4=predict(fit4,newdata=dat[!train,],n.trees=300,type="response")
plot.roc(dat$nextchurn[!train],as.vector(yhat4),print.auc=TRUE,col=5,print.auc.y=.2)

##revised code
fit4 = randomForest(nextchurn~.-SUBSCRIPTIONID-train,data=rdat[rtrain,], n.tree = 300)
summary(fit4)
yhat4=predict(fit4,newdata=rdat[!rtrain,],n.trees=300,type="response")
plot.roc(rdat$nextchurn[!rtrain],as.vector(yhat4),print.auc=TRUE,col=5,print.auc.y=.2)
```

##3.6 GBM
```{r}
fit3=gbm(nextchurn~.-SUBSCRIPTIONID-train,data=dat[train,],interaction.depth=2,n.trees=3000,shrinkage=.01)
summary(fit3)
gbm.perf(fit3)
yhat3=predict(fit3,newdata=dat[!train,],n.trees=300,type="response")
length(yhat3)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)
```

##3.7 GAM
```{r}
fit5 = gam(nextchurn~s(REGULARITY)+s(NHOMEPAGE)+s(sessions)+s(thismon)+s(t)+s(PVs)+s(DevMobile)+s(DevTablet)+s(DevDesktop)+s(DevApp)+s(SrcSearch)+s(SrcSocial)+s(MktOutside)
           +s(TopicBreakNews)+s(TopicLocalCom)+s(TopicNatWorld)+s(TopicLocGov)+s(TopicStateGov)+s(TopicNatGov)+s(TopicHealth)+s(TopicCrime)+s(TopicElect)+s(TopicColSport)+s(TopicProSport)
           +s(TopicHSsport)+s(TopicFireAccident)+s(TopicImmigration)+s(TopicEduc)+s(TopicOpenClose)+s(TopicJobEcon)+s(TopicLocBus)+s(TopicRealEstate)+s(TopicWeather)+s(TopicEnviron)+s(TopicRestDine)
           +s(TopicEvent)+s(TopicCelebrity)+s(TopicEntertain)+s(TopicTourism)+s(TopicTraffic)+s(NewsLetSubs)+s(NewsLetUnsubs) + s(totsubs), data = dat, subset = (train==1))
summary(fit5)
```




#4 Full Model with Transformation
##4.1 Logistic Regression
```{r}
fit1 = glm(nextchurn ~REGULARITY + sqrt(REGULARITY) +log(REGULARITY+1)+I(REGULARITY^2)
           + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE+1)+ I(NHOMEPAGE^2) 
           + sessions + sqrt(sessions) +log(sessions+1)+ I(sessions^2)
           + thismon  + sqrt(thismon) +log(thismon+1)+ I(thismon^2)
           + t + sqrt(t) +log(t+1)+ I(t^2)
           + PVs + sqrt(PVs) +log(PVs+1)+ I(PVs^2)
           + TopicBreakNews + sqrt(TopicBreakNews) +log(TopicBreakNews+1)+ I(TopicBreakNews^2)
           +TopicLocalCom + sqrt(TopicLocalCom) +log(TopicLocalCom+1)+ I(TopicLocalCom^2)
           +TopicNatWorld+ sqrt(TopicNatWorld) +log(TopicNatWorld+1)+ I(TopicNatWorld^2)
           +TopicLocGov+ sqrt(TopicLocGov) +log(TopicLocGov+1)+ I(TopicLocGov^2)
           +TopicStateGov+ sqrt(TopicStateGov) +log(TopicStateGov+1)+ I(TopicStateGov^2)
           +TopicNatGov+ sqrt(TopicNatGov) +log(TopicNatGov+1)+ I(TopicNatGov^2)
           +TopicHealth+ sqrt(TopicHealth) +log(TopicHealth+1)+ I(TopicHealth^2)
           +TopicCrime+ sqrt(TopicCrime) +log(TopicCrime+1)+ I(TopicCrime^2)
           +TopicElect+ sqrt(TopicElect) +log(TopicElect+1)+ I(TopicElect^2)
           +TopicCrime+sqrt(TopicCrime)+log(TopicCrime+1)+ I(TopicCrime^2)
           +TopicColSport+ sqrt(TopicColSport) +log(TopicColSport+1)+ I(TopicColSport^2)
           +TopicProSport+ sqrt(TopicProSport) +log(TopicProSport+1)+ I(TopicProSport^2)
           +TopicHSsport+ sqrt(TopicHSsport) +log(TopicHSsport+1)+ I(TopicHSsport^2)
           +TopicFireAccident+ sqrt(TopicFireAccident) +log(TopicFireAccident+1)+ I(TopicFireAccident^2)
           +TopicImmigration+ sqrt(TopicImmigration) +log(TopicImmigration+1)+ I(TopicImmigration^2)
           +TopicEduc+ sqrt(TopicEduc) +log(TopicEduc+1)+ I(TopicEduc^2)
           +TopicOpenClose+ sqrt(TopicOpenClose) +log(TopicOpenClose+1)+ I(TopicOpenClose^2)
           +TopicJobEcon+ sqrt(TopicJobEcon) +log(TopicJobEcon+1)+ I(TopicJobEcon^2)
           +TopicLocBus+ sqrt(TopicLocBus) +log(TopicLocBus+1)+ I(TopicLocBus^2)
           +TopicRealEstate+ sqrt(TopicRealEstate) +log(TopicRealEstate+1)+ I(TopicRealEstate^2)
           +TopicWeather + sqrt(TopicWeather) +log(TopicWeather+1)+ I(TopicWeather^2)
           +TopicEnviron + sqrt(TopicEnviron) +log(TopicEnviron+1)+ I(TopicEnviron^2)
           +TopicRestDine + sqrt(TopicRestDine) +log(TopicRestDine+1)+ I(TopicRestDine^2)
           +TopicEvent + sqrt(TopicEvent) +log(TopicEvent+1)+ I(TopicEvent^2)
           +TopicCelebrity + sqrt(TopicCelebrity) +log(TopicCelebrity+1)+ I(TopicCelebrity^2)
           +TopicEntertain + sqrt(TopicEntertain) +log(TopicEntertain+1)+ I(TopicEntertain^2)
           +TopicTourism + sqrt(TopicTourism) +log(TopicTourism+1)+ I(TopicTourism^2)
           +TopicTraffic + sqrt(TopicTraffic) +log(TopicTraffic+1)+ I(TopicTraffic^2)
           +SrcSearch + sqrt(SrcSearch) +log(SrcSearch+1)+ I(SrcSearch^2)
           +SrcSocial + sqrt(SrcSocial) +log(SrcSocial+1)+ I(SrcSocial^2)
           +MktOutside + sqrt(MktOutside) +log(MktOutside+1)+ I(MktOutside^2)
           +DevMobile + sqrt(DevMobile) +log(DevMobile+1)+ I(DevMobile^2)
           +DevTablet + sqrt(DevTablet) +log(DevTablet+1)+ I(DevTablet^2)
           +DevDesktop + sqrt(DevDesktop) +log(DevDesktop+1)+ I(DevDesktop^2)
           +DevApp + sqrt(DevApp) +log(DevApp+1)+ I(DevApp^2)
           +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
           +NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs+1)+ I(NewsLetUnsubs^2)
           +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
           +totsubs+sqrt(totsubs)+log(totsubs+1)+I(totsubs^2), binomial, dat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot.roc(dat$nextchurn[!train],yhat,print.auc=TRUE)

##4.2 Ridge
x = model.matrix(nextchurn ~ REGULARITY + sqrt(REGULARITY) +log(REGULARITY)+I(REGULARITY^2)
                 + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE)+ I(NHOMEPAGE^2) + sessions + sqrt(sessions) +log(sessions)+ I(sessions^2)+ thismon  + sqrt(thismon) +log(thismon)+ I(thismon^2)+ t + sqrt(t) +log(t)+ I(t^2)+ PVs + sqrt(PVs) +log(PVs)+ I(PVs^2)+ TopicBreakNews + sqrt(TopicBreakNews) +log(TopicBreakNews)+ I(TopicBreakNews^2)+TopicLocalCom + sqrt(TopicLocalCom) +log(TopicLocalCom)+ I(TopicLocalCom^2)+TopicNatWorld+ sqrt(TopicNatWorld) +log(TopicNatWorld)+ I(TopicNatWorld^2)+TopicLocGov+ sqrt(TopicLocGov) +log(TopicLocGov)+ I(TopicLocGov^2)+TopicStateGov+ sqrt(TopicStateGov) +log(TopicStateGov)+ I(TopicStateGov^2)+TopicNatGov+ sqrt(TopicNatGov) +log(TopicNatGov)+ I(TopicNatGov^2)+TopicHealth+ sqrt(TopicHealth) +log(TopicHealth)+ I(TopicHealth^2)+TopicCrime+ sqrt(TopicCrime) +log(TopicCrime)+ I(TopicCrime^2)+TopicElect+ sqrt(TopicElect) +log(TopicElect)+ I(TopicElect^2)+log(TopicCrime)+ I(TopicCrime^2)+TopicColSport+ sqrt(TopicColSport) +log(TopicColSport )+ I(TopicColSport^2)+TopicProSport+ sqrt(TopicProSport) +log(TopicProSport)+ I(TopicProSport^2)+TopicHSsport+ sqrt(TopicHSsport) +log(TopicHSsport)+ I(TopicHSsport^2)+TopicFireAccident+ sqrt(TopicFireAccident) +log(TopicFireAccident)+ I(TopicFireAccident^2)+TopicImmigration+ sqrt(TopicImmigration) +log(TopicImmigration)+ I(TopicImmigration^2)+TopicEduc+ sqrt(TopicEduc) +log(TopicEduc)+ I(TopicEduc^2)+TopicOpenClose+ sqrt(TopicOpenClose) +log(TopicOpenClose)+ I(TopicOpenClose^2)+TopicJobEcon+ sqrt(TopicJobEcon) +log(TopicJobEcon)+ I(TopicJobEcon^2)+TopicLocBus+ sqrt(TopicLocBus) +log(TopicLocBus)+ I(TopicLocBus^2)+TopicRealEstate+ sqrt(TopicRealEstate) +log(TopicRealEstate)+ I(TopicRealEstate^2)+TopicWeather + sqrt(TopicWeather) +log(TopicWeather)+ I(TopicWeather^2)+TopicEnviron + sqrt(TopicEnviron) +log(TopicEnviron)+ I(TopicEnviron^2)+TopicRestDine + sqrt(TopicRestDine) +log(TopicRestDine)+ I(TopicRestDine^2)+TopicEvent + sqrt(TopicEvent) +log(TopicEvent)+ I(TopicEvent^2)+TopicCelebrity + sqrt(TopicCelebrity) +log(TopicCelebrity)+ I(TopicCelebrity^2)+TopicEntertain + sqrt(TopicEntertain) +log(TopicEntertain)+ I(TopicEntertain^2)+TopicTourism + sqrt(TopicTourism) +log(TopicTourism)+ I(TopicTourism^2)+TopicTraffic + sqrt(TopicTraffic) +log(TopicTraffic)+ I(TopicTraffic^2)+SrcSearch + sqrt(SrcSearch) +log(SrcSearch)+ I(SrcSearch^2)+SrcSocial + sqrt(SrcSocial) +log(SrcSocial)+ I(SrcSocial^2)+MktOutside + sqrt(MktOutside) +log(MktOutside)+ I(MktOutside^2)+DevMobile + sqrt(DevMobile) +log(DevMobile)+ I(DevMobile^2)+DevTablet + sqrt(DevTablet) +log(DevTablet)+ I(DevTablet^2)+DevDesktop + sqrt(DevDesktop) +log(DevDesktop)+ I(DevDesktop^2)+DevApp + sqrt(DevApp) +log(DevApp)+ I(DevApp^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs)+ I(NewsLetUnsubs^2)+NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs)+ I(NewsLetSubs^2)+totsubs+sqrt(totsubs)+log(totsubs)+I(totsubs^2),dat)

fit.ridge = glmnet(x[train,], dat$nextchurn[train], alpha=0) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=0) 
yhat=predict(fit.ridge, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)
```

##4.3 Lasso
fit.lasso = glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv = cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat),print.auc=TRUE)


##4.4 Tree
fit2 = tree(nextchurn ~REGULARITY + sqrt(REGULARITY) +log(REGULARITY+1)+I(REGULARITY^2)
            + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE+1)+ I(NHOMEPAGE^2) 
            + sessions + sqrt(sessions) +log(sessions+1)+ I(sessions^2)
            + thismon  + sqrt(thismon) +log(thismon+1)+ I(thismon^2)
            + t + sqrt(t) +log(t+1)+ I(t^2)
            + PVs + sqrt(PVs) +log(PVs+1)+ I(PVs^2)
            + TopicBreakNews + sqrt(TopicBreakNews) +log(TopicBreakNews+1)+ I(TopicBreakNews^2)
            +TopicLocalCom + sqrt(TopicLocalCom) +log(TopicLocalCom+1)+ I(TopicLocalCom^2)
            +TopicNatWorld+ sqrt(TopicNatWorld) +log(TopicNatWorld+1)+ I(TopicNatWorld^2)
            +TopicLocGov+ sqrt(TopicLocGov) +log(TopicLocGov+1)+ I(TopicLocGov^2)
            +TopicStateGov+ sqrt(TopicStateGov) +log(TopicStateGov+1)+ I(TopicStateGov^2)
            +TopicNatGov+ sqrt(TopicNatGov) +log(TopicNatGov+1)+ I(TopicNatGov^2)
            +TopicHealth+ sqrt(TopicHealth) +log(TopicHealth+1)+ I(TopicHealth^2)
            +TopicCrime+ sqrt(TopicCrime) +log(TopicCrime+1)+ I(TopicCrime^2)
            +TopicElect+ sqrt(TopicElect) +log(TopicElect+1)+ I(TopicElect^2)
            +TopicCrime+sqrt(TopicCrime)+log(TopicCrime+1)+ I(TopicCrime^2)
            +TopicColSport+ sqrt(TopicColSport) +log(TopicColSport+1)+ I(TopicColSport^2)
            +TopicProSport+ sqrt(TopicProSport) +log(TopicProSport+1)+ I(TopicProSport^2)
            +TopicHSsport+ sqrt(TopicHSsport) +log(TopicHSsport+1)+ I(TopicHSsport^2)
            +TopicFireAccident+ sqrt(TopicFireAccident) +log(TopicFireAccident+1)+ I(TopicFireAccident^2)
            +TopicImmigration+ sqrt(TopicImmigration) +log(TopicImmigration+1)+ I(TopicImmigration^2)
            +TopicEduc+ sqrt(TopicEduc) +log(TopicEduc+1)+ I(TopicEduc^2)
            +TopicOpenClose+ sqrt(TopicOpenClose) +log(TopicOpenClose+1)+ I(TopicOpenClose^2)
            +TopicJobEcon+ sqrt(TopicJobEcon) +log(TopicJobEcon+1)+ I(TopicJobEcon^2)
            +TopicLocBus+ sqrt(TopicLocBus) +log(TopicLocBus+1)+ I(TopicLocBus^2)
            +TopicRealEstate+ sqrt(TopicRealEstate) +log(TopicRealEstate+1)+ I(TopicRealEstate^2)
            +TopicWeather + sqrt(TopicWeather) +log(TopicWeather+1)+ I(TopicWeather^2)
            +TopicEnviron + sqrt(TopicEnviron) +log(TopicEnviron+1)+ I(TopicEnviron^2)
            +TopicRestDine + sqrt(TopicRestDine) +log(TopicRestDine+1)+ I(TopicRestDine^2)
            +TopicEvent + sqrt(TopicEvent) +log(TopicEvent+1)+ I(TopicEvent^2)
            +TopicCelebrity + sqrt(TopicCelebrity) +log(TopicCelebrity+1)+ I(TopicCelebrity^2)
            +TopicEntertain + sqrt(TopicEntertain) +log(TopicEntertain+1)+ I(TopicEntertain^2)
            +TopicTourism + sqrt(TopicTourism) +log(TopicTourism+1)+ I(TopicTourism^2)
            +TopicTraffic + sqrt(TopicTraffic) +log(TopicTraffic+1)+ I(TopicTraffic^2)
            +SrcSearch + sqrt(SrcSearch) +log(SrcSearch+1)+ I(SrcSearch^2)
            +SrcSocial + sqrt(SrcSocial) +log(SrcSocial+1)+ I(SrcSocial^2)
            +MktOutside + sqrt(MktOutside) +log(MktOutside+1)+ I(MktOutside^2)
            +DevMobile + sqrt(DevMobile) +log(DevMobile+1)+ I(DevMobile^2)
            +DevTablet + sqrt(DevTablet) +log(DevTablet+1)+ I(DevTablet^2)
            +DevDesktop + sqrt(DevDesktop) +log(DevDesktop+1)+ I(DevDesktop^2)
            +DevApp + sqrt(DevApp) +log(DevApp+1)+ I(DevApp^2)
            +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
            +NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs+1)+ I(NewsLetUnsubs^2)
            +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
            +totsubs+sqrt(totsubs)+log(totsubs+1)+I(totsubs^2),data=dat[train,],mindev=.0001)
fit2
plot(fit2, type="uniform")
fit.tree = prune.tree(fit2, best=100)
plot(fit.tree)
yhat2=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector")
plot.roc(dat$nextchurn[!train],as.vector(yhat2),print.auc=TRUE,col=5,print.auc.y=.2)
```

##4.5 RF
```{r}
fit4 = randomForest(nextchurn ~REGULARITY + sqrt(REGULARITY) +log(REGULARITY+1)+I(REGULARITY^2)
                    + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE+1)+ I(NHOMEPAGE^2) 
                    + sessions + sqrt(sessions) +log(sessions+1)+ I(sessions^2)
                    + thismon  + sqrt(thismon) +log(thismon+1)+ I(thismon^2)
                    + t + sqrt(t) +log(t+1)+ I(t^2)
                    + PVs + sqrt(PVs) +log(PVs+1)+ I(PVs^2)
                    + TopicBreakNews + sqrt(TopicBreakNews) +log(TopicBreakNews+1)+ I(TopicBreakNews^2)
                    +TopicLocalCom + sqrt(TopicLocalCom) +log(TopicLocalCom+1)+ I(TopicLocalCom^2)
                    +TopicNatWorld+ sqrt(TopicNatWorld) +log(TopicNatWorld+1)+ I(TopicNatWorld^2)
                    +TopicLocGov+ sqrt(TopicLocGov) +log(TopicLocGov+1)+ I(TopicLocGov^2)
                    +TopicStateGov+ sqrt(TopicStateGov) +log(TopicStateGov+1)+ I(TopicStateGov^2)
                    +TopicNatGov+ sqrt(TopicNatGov) +log(TopicNatGov+1)+ I(TopicNatGov^2)
                    +TopicHealth+ sqrt(TopicHealth) +log(TopicHealth+1)+ I(TopicHealth^2)
                    +TopicCrime+ sqrt(TopicCrime) +log(TopicCrime+1)+ I(TopicCrime^2)
                    +TopicElect+ sqrt(TopicElect) +log(TopicElect+1)+ I(TopicElect^2)
                    +TopicCrime+sqrt(TopicCrime)+log(TopicCrime+1)+ I(TopicCrime^2)
                    +TopicColSport+ sqrt(TopicColSport) +log(TopicColSport+1)+ I(TopicColSport^2)
                    +TopicProSport+ sqrt(TopicProSport) +log(TopicProSport+1)+ I(TopicProSport^2)
                    +TopicHSsport+ sqrt(TopicHSsport) +log(TopicHSsport+1)+ I(TopicHSsport^2)
                    +TopicFireAccident+ sqrt(TopicFireAccident) +log(TopicFireAccident+1)+ I(TopicFireAccident^2)
                    +TopicImmigration+ sqrt(TopicImmigration) +log(TopicImmigration+1)+ I(TopicImmigration^2)
                    +TopicEduc+ sqrt(TopicEduc) +log(TopicEduc+1)+ I(TopicEduc^2)
                    +TopicOpenClose+ sqrt(TopicOpenClose) +log(TopicOpenClose+1)+ I(TopicOpenClose^2)
                    +TopicJobEcon+ sqrt(TopicJobEcon) +log(TopicJobEcon+1)+ I(TopicJobEcon^2)
                    +TopicLocBus+ sqrt(TopicLocBus) +log(TopicLocBus+1)+ I(TopicLocBus^2)
                    +TopicRealEstate+ sqrt(TopicRealEstate) +log(TopicRealEstate+1)+ I(TopicRealEstate^2)
                    +TopicWeather + sqrt(TopicWeather) +log(TopicWeather+1)+ I(TopicWeather^2)
                    +TopicEnviron + sqrt(TopicEnviron) +log(TopicEnviron+1)+ I(TopicEnviron^2)
                    +TopicRestDine + sqrt(TopicRestDine) +log(TopicRestDine+1)+ I(TopicRestDine^2)
                    +TopicEvent + sqrt(TopicEvent) +log(TopicEvent+1)+ I(TopicEvent^2)
                    +TopicCelebrity + sqrt(TopicCelebrity) +log(TopicCelebrity+1)+ I(TopicCelebrity^2)
                    +TopicEntertain + sqrt(TopicEntertain) +log(TopicEntertain+1)+ I(TopicEntertain^2)
                    +TopicTourism + sqrt(TopicTourism) +log(TopicTourism+1)+ I(TopicTourism^2)
                    +TopicTraffic + sqrt(TopicTraffic) +log(TopicTraffic+1)+ I(TopicTraffic^2)
                    +SrcSearch + sqrt(SrcSearch) +log(SrcSearch+1)+ I(SrcSearch^2)
                    +SrcSocial + sqrt(SrcSocial) +log(SrcSocial+1)+ I(SrcSocial^2)
                    +MktOutside + sqrt(MktOutside) +log(MktOutside+1)+ I(MktOutside^2)
                    +DevMobile + sqrt(DevMobile) +log(DevMobile+1)+ I(DevMobile^2)
                    +DevTablet + sqrt(DevTablet) +log(DevTablet+1)+ I(DevTablet^2)
                    +DevDesktop + sqrt(DevDesktop) +log(DevDesktop+1)+ I(DevDesktop^2)
                    +DevApp + sqrt(DevApp) +log(DevApp+1)+ I(DevApp^2)
                    +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
                    +NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs+1)+ I(NewsLetUnsubs^2)
                    +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
                    +totsubs+sqrt(totsubs)+log(totsubs+1)+I(totsubs^2),data=dat[train,], n.tree = 300)
summary(fit4)
yhat4=predict(fit4,newdata=dat[!train,],n.trees=300,type="response")
plot.roc(dat$nextchurn[!train],as.vector(yhat4),print.auc=TRUE,col=5,print.auc.y=.2)
```

##4.6 GBM
```{r}
fit3=gbm(nextchurn ~REGULARITY + sqrt(REGULARITY) +log(REGULARITY+1)+I(REGULARITY^2)
         + NHOMEPAGE + + sqrt(NHOMEPAGE) +log(NHOMEPAGE+1)+ I(NHOMEPAGE^2) 
         + sessions + sqrt(sessions) +log(sessions+1)+ I(sessions^2)
         + thismon  + sqrt(thismon) +log(thismon+1)+ I(thismon^2)
         + t + sqrt(t) +log(t+1)+ I(t^2)
         + PVs + sqrt(PVs) +log(PVs+1)+ I(PVs^2)
         + TopicBreakNews + sqrt(TopicBreakNews) +log(TopicBreakNews+1)+ I(TopicBreakNews^2)
         +TopicLocalCom + sqrt(TopicLocalCom) +log(TopicLocalCom+1)+ I(TopicLocalCom^2)
         +TopicNatWorld+ sqrt(TopicNatWorld) +log(TopicNatWorld+1)+ I(TopicNatWorld^2)
         +TopicLocGov+ sqrt(TopicLocGov) +log(TopicLocGov+1)+ I(TopicLocGov^2)
         +TopicStateGov+ sqrt(TopicStateGov) +log(TopicStateGov+1)+ I(TopicStateGov^2)
         +TopicNatGov+ sqrt(TopicNatGov) +log(TopicNatGov+1)+ I(TopicNatGov^2)
         +TopicHealth+ sqrt(TopicHealth) +log(TopicHealth+1)+ I(TopicHealth^2)
         +TopicCrime+ sqrt(TopicCrime) +log(TopicCrime+1)+ I(TopicCrime^2)
         +TopicElect+ sqrt(TopicElect) +log(TopicElect+1)+ I(TopicElect^2)
         +TopicCrime+sqrt(TopicCrime)+log(TopicCrime+1)+ I(TopicCrime^2)
         +TopicColSport+ sqrt(TopicColSport) +log(TopicColSport+1)+ I(TopicColSport^2)
         +TopicProSport+ sqrt(TopicProSport) +log(TopicProSport+1)+ I(TopicProSport^2)
         +TopicHSsport+ sqrt(TopicHSsport) +log(TopicHSsport+1)+ I(TopicHSsport^2)
         +TopicFireAccident+ sqrt(TopicFireAccident) +log(TopicFireAccident+1)+ I(TopicFireAccident^2)
         +TopicImmigration+ sqrt(TopicImmigration) +log(TopicImmigration+1)+ I(TopicImmigration^2)
         +TopicEduc+ sqrt(TopicEduc) +log(TopicEduc+1)+ I(TopicEduc^2)
         +TopicOpenClose+ sqrt(TopicOpenClose) +log(TopicOpenClose+1)+ I(TopicOpenClose^2)
         +TopicJobEcon+ sqrt(TopicJobEcon) +log(TopicJobEcon+1)+ I(TopicJobEcon^2)
         +TopicLocBus+ sqrt(TopicLocBus) +log(TopicLocBus+1)+ I(TopicLocBus^2)
         +TopicRealEstate+ sqrt(TopicRealEstate) +log(TopicRealEstate+1)+ I(TopicRealEstate^2)
         +TopicWeather + sqrt(TopicWeather) +log(TopicWeather+1)+ I(TopicWeather^2)
         +TopicEnviron + sqrt(TopicEnviron) +log(TopicEnviron+1)+ I(TopicEnviron^2)
         +TopicRestDine + sqrt(TopicRestDine) +log(TopicRestDine+1)+ I(TopicRestDine^2)
         +TopicEvent + sqrt(TopicEvent) +log(TopicEvent+1)+ I(TopicEvent^2)
         +TopicCelebrity + sqrt(TopicCelebrity) +log(TopicCelebrity+1)+ I(TopicCelebrity^2)
         +TopicEntertain + sqrt(TopicEntertain) +log(TopicEntertain+1)+ I(TopicEntertain^2)
         +TopicTourism + sqrt(TopicTourism) +log(TopicTourism+1)+ I(TopicTourism^2)
         +TopicTraffic + sqrt(TopicTraffic) +log(TopicTraffic+1)+ I(TopicTraffic^2)
         +SrcSearch + sqrt(SrcSearch) +log(SrcSearch+1)+ I(SrcSearch^2)
         +SrcSocial + sqrt(SrcSocial) +log(SrcSocial+1)+ I(SrcSocial^2)
         +MktOutside + sqrt(MktOutside) +log(MktOutside+1)+ I(MktOutside^2)
         +DevMobile + sqrt(DevMobile) +log(DevMobile+1)+ I(DevMobile^2)
         +DevTablet + sqrt(DevTablet) +log(DevTablet+1)+ I(DevTablet^2)
         +DevDesktop + sqrt(DevDesktop) +log(DevDesktop+1)+ I(DevDesktop^2)
         +DevApp + sqrt(DevApp) +log(DevApp+1)+ I(DevApp^2)
         +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
         +NewsLetUnsubs + sqrt(NewsLetUnsubs) +log(NewsLetUnsubs+1)+ I(NewsLetUnsubs^2)
         +NewsLetSubs + sqrt(NewsLetSubs) +log(NewsLetSubs+1)+ I(NewsLetSubs^2)
         +totsubs+sqrt(totsubs)+log(totsubs+1)+I(totsubs^2),data=dat[train,],interaction.depth=2,n.trees=3000,shrinkage=.01)
summary(fit3)
gbm.perf(fit3)
yhat3=predict(fit3,newdata=dat[!train,],n.trees=300,type="response")
length(yhat3)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)

##4.7 GAM
```{r}







#Best Model
#5.1
fit1 = glm(nextchurn~log(thismon+1) #power predictor
           + sqrt(t) 
           + sqrt(PVs) 
           + sqrt(NewsLetSubs) #power predictor
           + sqrt(NewsLetUnsubs)
           + sqrt(totsubs), binomial, dat[train,])
summary(fit1)
yhat = predict(fit1, dat[!train,]) 
plot <- plot.roc(dat$nextchurn[!train],yhat,col=1,legacy.axes=T,print.auc=TRUE,print.auc.col=1)


##5.3 Lasso
x = model.matrix(nextchurn~thismon #power predictor
                 + t 
                 + PVs 
                 + NewsLetSubs #power predictor
                 + NewsLetUnsubs
                 + totsubs, dat)
fit.lasso=glmnet(x[train,], dat$nextchurn[train], alpha=1) 
fit.cv=cv.glmnet(x[train,], dat$nextchurn[train], alpha=1) 
yhat2=predict(fit.lasso, s=fit.cv$lambda.min, newx=x[!train,]) 
plot.roc(dat$nextchurn[!train],as.vector(yhat2),add=T,col=3,print.auc=TRUE, print.auc.x=.3, print.auc.y=.8,print.auc.col=3)





## 5.4 TREE
## WITH TRANSFORMATION
fit3=tree(nextchurn~log(thismon+1) #power predictor
         + sqrt(t) 
         + sqrt(PVs) 
         + sqrt(NewsLetSubs) #power predictor
         + sqrt(NewsLetUnsubs)
         + sqrt(totsubs),data=dat[train,],mindev=.0001)
fit3
summary(fit3)
#plot(fit3, type="uniform")
fit.tree = prune.tree(fit3, best=100)
#plot(fit.tree); fit.tree
yhat3=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector")
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)

## WITHOUT TRANSFORMATION
fit3=tree(nextchurn~thismon #power predictor
          + t 
          + PVs 
          + NewsLetSubs #power predictor
          + NewsLetUnsubs
          + totsubs,data=dat[train,],mindev=.0001)
fit3
summary(fit3)
#plot(fit3, type="uniform")
fit.tree = prune.tree(fit3, best=100)
#plot(fit.tree); fit.tree
yhat3=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector")
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)


#5.5  RANDOM FOREST (WITHOUT TRANSFORAMTION)
fit5=randomForest(nextchurn~thismon #power predictor
                  + t 
                  + PVs 
                  + NewsLetSubs #power predictor
                  + NewsLetUnsubs
                  + totsubs,data=dat[train,],n.trees=300)
summary(fit5)
yhat5=predict(fit5,newdata=dat[!train,],n.trees=300,type="response")
length(yhat5)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat5),print.auc=TRUE,col=5,print.auc.y=.2) #0.899


### GAINS TABLE
#5.5  RANDOM FOREST
fit5=randomForest(nextchurn~thismon #power predictor
                  + t 
                  + PVs 
                  + NewsLetSubs #power predictor
                  + NewsLetUnsubs
                  + totsubs,data=dat[train,],n.trees=300)
summary(fit5)
yhat5=predict(fit5,newdata=dat[!train,],n.trees=300,type="response")
length(yhat5)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat5),print.auc=TRUE,col=5,print.auc.y=.2) #0.899

## GAINS TABLE
gains = function(yhat5, nextchurn, ngrp=10){
  ans = data.frame(nextchurn=nextchurn, qtile=
                     cut(yhat5, breaks=quantile(yhat5, probs=seq(0,1,1/ngrp)), 
                         labels=paste("Q",ngrp:1, sep=""), include.lowest = T)
  ) %>%
    group_by(qtile) %>%
    summarise(n=n(), Nrechurn=sum(nextchurn),
              RespRate=Nrechurn/n) %>%
    arrange(desc(qtile)) %>%
    mutate(CumN=cumsum(n), CumResp=cumsum(Nrechurn),
           CumRespRate=CumResp/CumN)
  ans %>% mutate(liftResp=CumRespRate/CumRespRate[nrow(ans)])
}
gains(yhat5, dat$nextchurn[!train])

##1.5 GAINS TABLE
fit3=tree(nextchurn~log(thismon+1) #power predictor
          + sqrt(t) 
          + sqrt(PVs) 
          + sqrt(NewsLetSubs) #power predictor
          + sqrt(NewsLetUnsubs)
          + sqrt(totsubs),data=dat[train,],mindev=.0001)
fit3
summary(fit3)
yhat3=predict(fit.tree,newdata=dat[!train,],n.trees=300,type="vector")
plot.roc(dat$nextchurn[!train],as.vector(yhat3),legacy.axes=T,print.auc=TRUE,col=5,print.auc.y=.2) #0.885

gains = function(yhat3, nextchurn, ngrp=10){
  ans = data.frame(nextchurn=nextchurn, qtile=
                     cut(yhat3, breaks=quantile(yhat4, probs=seq(0,1,1/ngrp)), 
                         labels=paste("Q",ngrp:1, sep=""), include.lowest = T)
  ) %>%
    group_by(qtile) %>%
    summarise(n=n(), Nrechurn=sum(nextchurn),
              RespRate=Nrechurn/n) %>%
    arrange(desc(qtile)) %>%
    mutate(CumN=cumsum(n), CumResp=cumsum(Nrechurn),
           CumRespRate=CumResp/CumN)
  ans %>% mutate(liftResp=CumRespRate/CumRespRate[nrow(ans)])
}
gains(yhat4, dat$nextchurn[!train])

### example
gains = function(yhat, respond, amt, ngrp=5){
  ans = data.frame(amt=amt, respond=respond, qtile=
                     cut(yhat, breaks=quantile(yhat, probs=seq(0,1, 1/ngrp)), labels=paste("Q",ngrp:1, sep=""), include.lowest = T)
  ) %>%
    group_by(qtile) %>%
    summarise(n=n(), Nrespond=sum(respond), amt=sum(amt),
              RespRate=Nrespond/n, AvgAmt=amt/n) %>%
    arrange(desc(qtile)) %>%
    mutate(CumN=cumsum(n), CumResp=cumsum(Nrespond), CumAmt=cumsum(amt),
           CumRespRate=CumResp/CumN, CumAvgAmt=CumAmt/CumN)
  ans %>% mutate(liftResp=CumRespRate/CumRespRate[nrow(ans)],
                 liftAmt=CumAvgAmt/CumAvgAmt[nrow(ans)]) 
}

###
gains = function(yhat, respond, ngrp=10){
  ans = data.frame(respond=respond, qtile=
                     cut(yhat, breaks=quantile(yhat, probs=seq(0,1,1/ngrp)), 
                         labels=paste("Q",ngrp:1, sep=""), include.lowest = T)
  ) %>%
    group_by(qtile) %>%
    summarise(n=n(), Nrespond=sum(respond),
              RespRate=Nrespond/n) %>%
    arrange(desc(qtile)) %>%
    mutate(CumN=cumsum(n), CumResp=cumsum(Nrespond),
           CumRespRate=CumResp/CumN)
  ans %>% mutate(liftResp=CumRespRate/CumRespRate[nrow(ans)]) 
}
gains(yhat, dat$nextchurn[!train])

##
gains(yhat4, dat$nextchurn[!train])








```
##### IGNORE HERE
## WITH REGULARITY
fit3=gbm(nextchurn~.+ REGULARITY
         +log(thismon+1) #power predictor
         + sqrt(t) 
         + sqrt(PVs) 
         + sqrt(NewsLetSubs) #power predictor
         + sqrt(NewsLetUnsubs)
         + sqrt(totsubs),data=dat[train,],interaction.depth=2,n.trees=3000,shrinkage=.01)
summary(fit3)
yhat3=predict(fit3,newdata=dat[!train,],n.trees=300,type="response")
length(yhat3)
length(dat$nextchurn[!train])
plot.roc(dat$nextchurn[!train],as.vector(yhat3),print.auc=TRUE,col=5,print.auc.y=.2)
```


