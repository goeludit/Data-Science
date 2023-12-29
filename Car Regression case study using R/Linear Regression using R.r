

mydata<-read.csv("C:\\Users\\Udit Goel\\Desktop\\Files\\Projects\\ML DS\\Car Sales (R)\\Car_sales.csv")


# user written function for creating descriptive statistic
mystats <- function(x) {
  nmiss<-sum(is.na(x))
  a <- x[!is.na(x)]
  m <- mean(a)
  n <- length(a)
  s <- sd(a)
  min <- min(a)
  p1<-quantile(a,0.01)
  p5<-quantile(a,0.05)
  p10<-quantile(a,0.10)
  q1<-quantile(a,0.25)
  q2<-quantile(a,0.5)
  q3<-quantile(a,0.75)
  p90<-quantile(a,0.90)
  p95<-quantile(a,0.95)
  p99<-quantile(a,0.99)
  max <- max(a)
  UC <- m+3*s
  LC <- m-3*s
  outlier_flag<- max>UC | min<LC
  return(c(n=n, nmiss=nmiss, outlier_flag=outlier_flag, mean=m, stdev=s,min = min, p1=p1,p5=p5,p10=p10,q1=q1,q2=q2,q3=q3,p90=p90,p95=p95,p99=p99,max=max, UC=UC, LC=LC ))
}

vars <- c( "Sales_in_thousands" , "X__year_resale_value" ,  "Price_in_thousands",   
           "Engine_size" , "Horsepower", "Wheelbase" , "Width" ,"Power_perf_factor" , "Length" , "Curb_weight" , 
           "Fuel_capacity", "Fuel_efficiency" )

diag_stats<-t(data.frame(apply(mydata[vars], 2, mystats)))
diag_stats

# Writing Summary stats to external file

write.csv(diag_stats, file = "diag_stats.csv")

## OUTLIERS
mydata$Sales_in_thousands[mydata$Sales_in_thousands>257.086342425636] <-257.086342425636
mydata$X__year_resale_value[mydata$X__year_resale_value>52.4331275042866] <-52.4331275042866
mydata$Price_in_thousands[mydata$Price_in_thousands>70.4457144064253] <-70.4457144064253

## Missing value treatment
mydata<- mydata[!is.na(mydata$Sales_in_thousands),] # dropping obs where DV=missing
require(Hmisc)
mydata1<-data.frame(apply(mydata[vars],2, function(x) impute(x, mean))) #Imputing missings with mean for IV's
mydata1
mydat2<-cbind(mydata1,Vehicle_type=mydata$Vehicle_type )


#R code for categorical variables(Converting as factor variable)

mydat2$Vehicle_type <- factor(mydat2$Vehicle_type)
levels(mydat2$Vehicle_type) <- c("Car","Passenger")

hist(log(mydat2$Sales_in_thousands))
ln_sales<-log(mydat2$Sales_in_thousands)
require(car)
scatterplotMatrix(mydat2)

# Multiple Linear Regression Example 
fit <- lm(Sales_in_thousands ~ X__year_resale_value + Price_in_thousands+ Engine_size+Horsepower+Wheelbase+Width
          +Length+Curb_weight+Fuel_capacity+Fuel_efficiency+Vehicle_type, data=mydat2)

fit2 <- lm(ln_sales ~ X__year_resale_value + Price_in_thousands+ Engine_size+Wheelbase+Width
           +Length+Curb_weight+Fuel_capacity+Fuel_efficiency+Vehicle_type, data=mydat2)

summary(fit) # show results
summary(fit2)

hist(mydata1$Sales_in_thousands)
require(MASS)
step3<- stepAIC(fit2,direction="both")
ls(step3)
step3$anova

fit3<-lm(ln_sales~X__year_resale_value + Price_in_thousands + Engine_size + 
           Wheelbase +  Fuel_efficiency + 
           Vehicle_type, data = mydat2)

summary(fit3)

#Multicollinierity Check using VIF
library(car)
vif(fit3)



# Other useful functions 

coefficients(fit) # model coefficients
confint(fit, level=0.95) # CIs for model parameters 
fitted(fit) # predicted values
residuals(fit) # residuals
anova(fit) # anova table 
influence(fit) # regression diagnostics

#diagnostic plots 
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(fit)
 
#Creating dummy variables

mydata1$VT[mydata$Vehicle_type ==  "Passenger"] <- 1
mydata1$VT[mydata$Vehicle_type ==  "Car"] <- 0

summary(fit)

mydata1
############################### Scoring Data sets/Predicting the sales#####################################
mydata1$Ln_pre_sales<- (-2.321593  +
                          mydata1$Price_in_thousands* -0.054988 +
                          mydata1$Engine_size*0.254696  +
                          mydata1$Wheelbase*0.047546	+
                          mydata1$Fuel_efficiency*0.068975+
                          mydata1$VT*-0.573255)
mydata1$Pre_sales= exp(mydata1$Ln_pre_sales);

#################### Creating Deciles####################################
# find the decile locations 
decLocations <- quantile(mydata1$Pre_sales, probs = seq(0.1,0.9,by=0.1))

# use findInterval with -Inf and Inf as upper and lower bounds
mydata1$decile <- findInterval(mydata1$Pre_sales,c(-Inf,decLocations, Inf))

summary(mydata1$decile)
xtabs(~decile,mydata1)

write.csv(mydata1,"mydata1.csv")


##################################Decile Analysis Reports##

require(sqldf)
mydata1_DA <- sqldf("select decile, count(decile) as count, avg(Pre_sales) as avg_pre_sales,   avg(Sales_in_thousands) as sum_Actual_sales,
                     sum(Sales_in_thousands) as avg_Actual_sales                   
                           from mydata1
                           group by decile
                           order by decile desc")

write.csv(mydata1_DA,"mydata1_DA.csv")


###################################END OF REGRESSION case study 




