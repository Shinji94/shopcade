setwd('/Users/xinjiwang/Desktop/data')
library(dplyr)
normalize <- function(x) {
  return ((x - min(x)) / (max(x)-min(x))) }
data =read.csv('data1.csv')
data = filter(data,data$view != 0)
length(data$product_click)
plot(data$view,data$product_click)
head(data)

data2 = data.frame(data$item)
data2['widget_click'] = normalize(data$widget_click)
data2['view'] = normalize(data$view)
data2['load_time'] = normalize(data$load_time)
data2['productclick'] = normalize(data$product_click)
data2['click_perview'] = normalize(data$avg_pd_click)
data2['wid_click_perview'] = normalize(data$widget_click/data$view)

head(data2)
 
data2$score = normalize(( data$product_click*0.3
                           +data2$click_perview*0.5
                          +data2$widget_click*data2$productclick
                          -data2$view*0.1-0.1*data2$load_time))
# plot(data2$widget_click,data2$productclick)

ranking = data2[c(1,length(data2))]
ranking  = ranking[order(ranking$score,decreasing = T),]

for (i in 1 : dim(ranking[1])[1]){
     x = rnorm(1,1,0.1)
     ranking$random_score[i] = ranking$score[i]*x
}
ranking$random_score = normalize(ranking$random_score)
write.csv(ranking,'ranking.csv',row.names=FALSE)
write.csv(ranking,'data2.csv',row.names=FALSE)

n = round(length(data2$score)*0.1)
plot(data2$view[1:n],data2$score[1:n],col = 'red')
points(data2$view[n:length(data2$score)],data2$score[n:length(data2$score)],col = 'blue')

plot(data2$view[1:n],data2$productclick[1:n],col = 'red')
points(data2$view[n:length(data2$score)],data2$productclick[n:length(data2$score)],col = 'blue')


