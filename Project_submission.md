---
title: "Submission"
author: "Memed"
date: "22 février 2015"
output: html_document
---
** Predict the test set and output results for automatic grader.**
setwd("C:/Data_Scientist/Cours/Practical Machine Learning/Peer Assesment")

```r
# final model
rfFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, importance = ..1) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 2.52%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3870    9   14   10    3  0.00921659
## B   42 2573   38    1    4  0.03197893
## C    5   35 2331   22    3  0.02712855
## D    3    2  104 2138    5  0.05062167
## E    3   10   20   13 2479  0.01821782
```

```r
# prediction
testing_PC <- predict(preProc, testing[,-ncol(trainData)])
prediction <- as.character(predict(rfFit, testing_PC))
length(prediction)
```

```
## [1] 20
```
## write prediction files


```r
if(!file.exists("prediction")){dir.create("prediction")}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./prediction/problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}
pml_write_files(prediction)
```
