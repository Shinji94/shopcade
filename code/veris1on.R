library(dplyr)
normalize <- function(x) {
  return ((x - min(x)) / (max(x)-min(x))) }
setwd('/Users/xinjiwang/Desktop/data')
data =read.csv('1monthdata.csv')
data = filter(data,data$view !=0)

plot(data$view,data$click)

data$conersion = data$click/data$view
data$c_v = data$click/data$view
data$h_v = data$hover/data$view

globalmean = (sum(data$click)*12+sum(data$view)*1+sum(data$hover)*5)/(sum(data$click)+sum(data$view)+sum(data$hover))
Strenth = 30000
data$score = (data$click*5+data$hover*3+data$view*1+Strenth*globalmean)/(data$click+data$view+data$hover+Strenth)
score = data[c(3,length(data))]
score = score[order(score$score,decreasing = T),]
write.csv(data,'monthly.csv',row.names =FALSE)

globalmean2 = (sum(data$c_v)*10+sum(data$h_v)*5)/(sum(data$click)+sum(data$hover))
s2 = 6000
data$score2 =(data$c_v*5+data$h_v*3+s2*globalmean)/(data$c_v+data$h_v+s2) 

popular = filter(data,data$view>30)
result = popular[popular$pid,popular$score]
write.csv(result,'result.csv',rwo.names = FALSE)


