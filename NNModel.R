
###input the data sets, including normalised features and output
df = read.csv("datasets.csv")
str(df)

f1 = df$LIMIT_BAL
f2 = df$SEX
f3 = df$EDUCATION
f4 = df$MARRIAGE
f5 = df$AGE
f6 = df$PAY_0
f7 = df$PAY_2
f8 = df$PAY_3
f9 = df$PAY_4
f10 = df$PAY_5
f11 = df$PAY_6
f12 = df$BILL_AMT1
f13 = df$BILL_AMT2
f14 = df$BILL_AMT3
f15 = df$BILL_AMT4
f16 = df$BILL_AMT5
f17 = df$BILL_AMT6
f18 = df$PAY_AMT1
f19 = df$PAY_AMT2
f20 = df$PAY_AMT3
f21 = df$PAY_AMT4
f22 = df$PAY_AMT5
f23 = df$PAY_AMT6
output = df$default.payment.next.month

###split data sets into two parts: 80% for model train, 20% for model validation
train_df<- df[1:24000,]
validate_df<- df[24001:30000,]


library(neuralnet)
### use neural network algorithm to build the predictive model
### use 80% data sets for model train
NN_model <- neuralnet(output ~ f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13
                     +f14+f15+f16+f17+f18+f19+f20+f21+f22+f23,
                     train_df,hidden = 15, threshold=0.1,rep=10,algorithm="sag",learningrate=0.05,
                     err.fct="sse",act.fct="logistic",lifesign = "full",linear.output = FALSE)

### use 20% data sets for model validation
NN_model_output <- compute(NN_model, validate_df[,1:23])
str(NN_model_output)
forecasting_result <- ifelse(NN_model_output$net.result > 0.5,1,0)
str(forecasting_result)
error <- mean(forecasting_result != validate_df[,24])
print(paste('Accuracy',1-error))

###[1] "Accuracy 0.839"
NN_model_output <- compute(NN_model, validate_df[,1:23])
str(NN_model_output)
out <- data.frame(NN_model_output$net.result,validate_df[,24])
write.csv(out, file = "out.csv")


