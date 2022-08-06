# Sys.setlocale(category="LC_ALL",locale = "English_United States.1252")
ssd <- read.csv("datasets/SSD dataset updated.csv")
# mean(ssd[['price']])
# sd(ssd[['price']])
# boxplot(ssd[['price']],ylab="Price")
a <- ssd[['price']]
b <- ssd[['capacity']]
c <- a/b
plot(x=ssd[['read']],y=c,main='PPG against reading speed',xlab='Read speed (MB / sec)',ylab = 'price / capacity (GB)')

