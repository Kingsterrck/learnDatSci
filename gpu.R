data <- read.csv("datasets/GPUMarkPrice.csv");
price <- data[['currentPrice']]
perf <- data[['timeSpy']]
plot(x=price,y=perf)