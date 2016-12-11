library(ggplot2)

parse<-function(graph)
{
  graph$V1 = as.numeric(substring(graph$V1, 3,nchar(as.character(graph$V1))))
  graph[,ncol(graph)] = substring(graph[,ncol(graph)], 1, nchar(as.character(graph[,ncol(graph)]))-2)
  return (graph)
}

setwd("C:/Users/Akhil Prakash/Desktop/Stanford/CS 224w/CS224_Project/final out directory/InstagramWithContentBasedDefaultRecommender.RandomPoliticalPreference.IndividualPLike.CollaborationFriendGraph")
  #"C:/Users/Akhil Prakash/Desktop/Stanford/CS 224w/CS224_Project/out/InstagramWithRandomDefaultRecommender.RandomNullRecommender.FriendsPoliticalPreference.EmpiricalPLike.500NumOnline.20NumRecs.100Iterations")
  #"C:/Users/Akhil Prakash/Desktop/Stanford/CS 224w/CS224_Project/out/CollaborativeFilteringRecommender.RandomNullRecommender.PropagationPoliticalPreference.EmpiricalPLike.500NumOnline.20NumRecs.32Iterations")
#"C:/Users/Akhil Prakash/Desktop/Stanford/CS 224w/CS224_Project/out/CollaborativeFilteringRecommender.RandomNullRecommender.PropagationPoliticalPreference.EmpiricalPLike.500NumOnline.20NumRecs.32Iterations"
# has analysis done for this "C:/Users/Akhil Prakash/Desktop/Stanford/CS 224w/CS224_Project/out/CollaborativeFilteringRecommender.RandomNullRecommender.PropagationPoliticalPreference.EmpiricalPLike.500NumOnline.20NumRecs.32Iterations"
#"C:/Users/Akhil Prakash/Desktop/Stanford/CS 224w/CS224_Project/CollaborativeFilteringRecommender.RandomNullRecommender.PropagationPoliticalPreference.EmpiricalPLike.500NumOnline.20NumRecs.50Iterations"

stackedBar = parse(read.csv("stackedBar.csv", header = F))

colnames(stackedBar) <- c("articleId", "type", "Source")
stackedBar$type = as.factor(stackedBar$type)

stackedBar <- within(stackedBar, 
                   articleId <- factor(articleId, 
                                      levels=names(sort(table(articleId), 
                                                        decreasing=TRUE))))

stackedBar$type = factor(stackedBar$type, levels = 2:-2)

ggplot(data = stackedBar) + geom_bar(aes(x = articleId, fill = type), position = "fill") + xlab("Article Id") + ylab("Proportion") + ggtitle("Proportion of User Polticalness for each Article") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggplot(data = stackedBar) + geom_bar(aes(x = articleId, fill = type)) + xlab("Article Id") + ylab("Frequency") + ggtitle("User Polticalness who Like each Article") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

first50 = names(sort(table(stackedBar$articleId), decreasing=TRUE))[1:30]

stackedBarSubset = subset(stackedBar, articleId %in% first50)

ggplot(data = stackedBarSubset) + geom_bar(aes(x = articleId, fill = type), position = "fill") +
  xlab("Article Id") + ylab("Proportion") + ggtitle("Proportion of User Polticalness for each Article") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggplot(data = stackedBarSubset) + geom_bar(aes(x = articleId, fill = type)) +
  xlab("Article Id") + ylab("Frequency") + ggtitle("User Polticalness who Like each Article") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

#test if these poropations match up witht he proapotions of users then knwo basically eveyrone reaidng anything

#for the most liked articles on arge ahve 2: 23 %
#                                         1: 30%
#                                         0: 25 %
#                                        -1: 20%
#                                        -2: 5% with lots of outlisers that are poropably fox news

#-2 way off
#1 is off by .1
#2 is off by .1

barplot(table(stackedBar$Source[stackedBar$Source %in% stackedBarSubset$Source])/length(stackedBar$Source), main = "subset")
barplot(table(stackedBarSubset$Source)/length(stackedBarSubset$Source), main = "non subset")

#ABC news articles like disbutriton about symmetric
#If subset larger than non subset, then skewed disitbrution of poeple liking that source
#NBC news skewed right like disbturtion
#CNN skewed right like disitrbution
#Gelnn Beck skewed left like disitrbution
#MSNBC sliltyt skewd left like disitbrution


ggplot(data = stackedBar) + geom_bar(aes(x = Source))
ggplot(data = stackedBarSubset) + geom_bar(aes(x = Source))

ggplot(data = stackedBarSubset) + geom_bar(aes(x = Source, fill = type), position = "fill") +
  ylab("Proportion") + ggtitle("User Polticalness based on Source")
ggplot(data = stackedBarSubset) + geom_bar(aes(x = Source, fill = type)) +
  ylab("Frequency") + ggtitle("User Polticalness based on Source")

#Fox News, Glen Beck, and Hannity are causing the large about of -2 likes in the top 30

location = locator()

total = location$y[length(location$y)] - location$y[1]
SUM =0
for (i in 2:length(location$y))
{
  print( (location$y[i] - location$y[i-1])/total)
  SUM = SUM + (location$y[i] - location$y[i-1])/total
}

print(SUM)

#most of the highly liked articles have roughly the same spread. 
N =  1534+ 1198+ 1116+ 897+ 497
pneg1 = 1534/N
pneg2 = 1116/N
pzero = 1198/N
ppos1 = 897/N
ppos2 = 497/N

#clacualte stdev for each article
articleIdStdevMat = matrix(0, nrow = length(unique(stackedBar$articleId)), ncol = 3)
idx = 1
for (aId in unique(stackedBar$articleId))
{
  sub = subset(stackedBar, articleId == aId)
  articleIdStdevMat[idx,] = c(aId, sd(as.numeric(sub$type)), nrow(sub))
  idx = idx + 1
}

plot(sort(articleIdStdevMat[,2]))

plot(articleIdStdevMat[,3], articleIdStdevMat[,2], xlab = "Number of LIkes", ylab = "Stdev")

#most articles that hav ehad hihg variance have low number of likes.

top30VarianceArticleIds = articleIdStdevMat[order(articleIdStdevMat[,2], decreasing = T),1][1:30]

top30VarianceArticle = articleIdStdevMat[order(articleIdStdevMat[,2], decreasing = T),][1:30,]

plot(top30VarianceArticle[,3], top30VarianceArticle[,2], xlab = "Number of LIkes", ylab = "Stdev")

stackedBarSubset = subset(stackedBar, articleId %in% top30VarianceArticleIds)

ggplot(data = stackedBarSubset) + geom_bar(aes(x = articleId, fill = type), position = "fill") +
  xlab("Article Id") + ylab("Proportion") + ggtitle("Proportion of User Polticalness for each Article") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
  
ggplot(data = stackedBarSubset) + geom_bar(aes(x = articleId, fill = type)) +
  xlab("Article Id") + ylab("Frequency") + ggtitle("User Polticalness Likes for each Article") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

#With instagram get a lot more 2 and -2 and less 0 so wil have more viariance so less polarization

#high vairiance artiles have disproprotioate number of 2 and -2 likeing them

#disporpotioate number of pos 2 people reading in comparisiont o porportion of pos 2 users.
#way less 0 poeple reading
#

ggplot(data = stackedBarSubset) + geom_bar(aes(x = Source))

table(stackedBarSubset$Source)/length(stackedBarSubset$Source)
table(stackedBar$Source[stackedBar$Source %in% stackedBarSubset$Source])/length(stackedBar$Source)

#IN instagram random, Polticio has very hihg variance. least polarizing

#Bloomberg ariticles have high variaince
#Economist aritlces have low vairance
#Guatrdian have low variaince
#HUffington Post have low vairance
#poltico have veyr low variaince
#THinRPogress* hav elow variaince
#Wall Street journal ahs low variaince

barplot(table(stackedBarSubset$Source)/length(stackedBarSubset$Source))
barplot(table(stackedBar$Source[stackedBar$Source %in% stackedBarSubset$Source])/length(stackedBar$Source))

#These plots ignore sample size
ggplot(data = stackedBarSubset) + geom_bar(aes(x = Source, fill = type), position = "fill") +
  xlab("Source") + ylab("Proportion") + ggtitle("Proportion of User Polticalness for each Article")
  
ggplot(data = stackedBarSubset) + geom_bar(aes(x = Source, fill = type)) +
  xlab("Source") + ylab("Frequency") + ggtitle("User Polticalness Likes for each Article")

#IF we ingore smaple size

#Mother Jones has lots of vairaince
#Sall strett journla nad BBc, Blookberg have little,
#THinkRPofess has lots of viariance
#BUzzfeed has some of variaince