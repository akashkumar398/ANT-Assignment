
library(readr)
train <- read_delim("train.csv", "~", escape_double = TRUE, 
                    trim_ws = TRUE)
#View(train)


summary(as.factor(train$Is_Response))

################
#Bad  Good 
#9605 20567
###############


unq_response <- as.data.frame(unique(train$Is_Response))
colnames(unq_response)[1] <- "Is_Response"
unq_response$level <- as.numeric(unq_response$Is_Response)
library(dplyr)

master_table_response <- left_join(x = train, y = unq_response)
train <- master_table_response

library(e1071)
library(text2vec)
library(data.table)
library(stringr)
library(caret)


train$Is_Response <- as.factor(train$Is_Response )

# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
set.seed(32984)
indexes <- createDataPartition(train$Is_Response, times = 1,
                               p = 0.7, list = FALSE)
#backup of train data
train1 <- train

#splitting
train <- train[indexes,]
test <- train[-indexes,]


#tokenization
prep_fun = tolower
tok_fun = word_tokenizer

#description#######
train_tokens = train$Description %>% 
  prep_fun %>% 
  tok_fun


################
it_train = itoken(train_tokens, 
                  progressbar = FALSE)
library(tm)
stop_words = tm::stopwords("en")
vocab = create_vocabulary(it_train, stopwords = stop_words, ngram = c(1L, 4L))
dim(vocab)
## To get the number of terms
range(vocab$doc_count)
range(vocab$term_count)
# To get the first 100 rows of words
vocab$term[1:99]

##############
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)


##################Browser###################################################
###########################

train_tokens1 = train$Browser_Used %>% 
  prep_fun %>% 
  tok_fun

################
it_train1 = itoken(train_tokens1, 
                  progressbar = FALSE)
vocab1 = create_vocabulary(it_train1, ngram = c(1L, 2L))
vectorizer1 = vocab_vectorizer(vocab1)

dtm_Browser = create_dtm(it_train1, vectorizer1)
# combine
#dtm_train_br = cbind(dtm_train, dtm_Browser)

###############################
##############################################################################



########################Device#############################################
###########################

train_tokens2 = train$Device_Used %>% 
  prep_fun %>% 
  tok_fun

################
it_train2 = itoken(train_tokens2, 
                   progressbar = FALSE)
vocab2 = create_vocabulary(it_train2)
vectorizer2 = vocab_vectorizer(vocab2)

dtm_device = create_dtm(it_train2, vectorizer2)


########################Device#############################################
###########################



# combine Description, Device and Browser
dtm_train_all = cbind(dtm_train, dtm_Browser, dtm_device)
dim(dtm_train_all)



###################################


library(xgboost)
label <- train$level - 1
xgmat <- xgb.DMatrix(dtm_train_all, label = label)


number_Of_Classes <- 2

# Save an object to a file
#saveRDS(number_Of_Classes, file = "number_Of_Classes.rds")
# Restore the object
#number_Of_Classes <- readRDS(file = "number_Of_Classes.rds")

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "bst:eta" = 0.3,
              "bst:gamma" = 0.5,
              "bst:min_child_weight" = 2,
              "bst:max_depth" = 10,
              "bst:alpha" = 2,
              "bst:silent" = 0,
              "verbose" = 1,
              "num_class" = number_Of_Classes)

nround    <- 300 # number of XGBoost rounds
cv.nfold  <- 5





cv_model <- xgb.cv(params = param,
                   data = xgmat, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE)





# Mutate xgb output to deliver hard predictions
model_output_cv <- as.data.frame(cv_model$pred) %>% mutate(max = max.col(., ties.method = "last"), label = label + 1)
model_output_cv$Description = train$Description
# View(xgb_train_preds)


confusionMatrix(factor(model_output_cv$max), 
                factor(train$level),
                mode = "everything")

bst_model <- xgb.train(params = param,
                       data = xgmat,
                       nrounds = 400,
                       verbose = TRUE)

summary(bst_model)


getwd()

xgb.save(bst_model, "Classification.model")
print ('finish training')

save(bst_model, file='xgboost_for_Class_prediction.RData')


require(xgboost)
require(methods)


View(test)
test_tokens1 = test$Description %>% 
  prep_fun %>% 
  tok_fun


it_test1 = itoken(test_tokens1, 
                 progressbar = FALSE)

dtm_test_desc = create_dtm(it_test1, vectorizer)



##################test Browser###################################################
###########################

test_tokens2 = test$Browser_Used %>% 
  prep_fun %>% 
  tok_fun

################
it_test2 = itoken(test_tokens2, 
                   progressbar = FALSE)
dtm_test_Browser = create_dtm(it_test2, vectorizer1)
###############################
##############################################################################



########################test Device#############################################
###########################

test_tokens3 = test$Device_Used %>% 
  prep_fun %>% 
  tok_fun

################
it_test3 = itoken(test_tokens3, 
                   progressbar = FALSE)

dtm_test_device = create_dtm(it_test3, vectorizer2)


########################Device#############################################
###########################



# combine test - Description, Device and Browser
dtm_test_all = cbind(dtm_test_desc, dtm_test_Browser, dtm_test_device)
dim(dtm_test_all)

xgmat_Pred <- xgb.DMatrix(dtm_test_all)

# Predict hold-out test set

preds_xgb <- predict(bst_model, xgmat_Pred)


head(preds_xgb)
head(cv_model$pred)

test_pred_res = as.data.frame(matrix(preds_xgb, ncol = 2, byrow=TRUE))


library(stringr)
# Mutate xgb output to deliver hard predictions
test_pred_res$review = test$Description
test_pred_res$Is_Response_actual = test$Is_Response
test_pred_res$max_prob = colnames(test_pred_res)[apply(test_pred_res[1:2],1,which.max)]
test_pred_res$level  <- substr(test_pred_res$max_prob, 2, nchar(test_pred_res$max_prob))


unq_response$level <- as.factor(unq_response$level)
test_pred_res$level <- as.factor(test_pred_res$level)

test_pred_res_f <- left_join(test_pred_res, unq_response,by = 'level' )

View(test_pred_res)
View(test_pred_res_f)



confusionMatrix(factor(test_pred_res_f$Is_Response), 
                factor(test$Is_Response),
                mode = "everything")


confusionMatrix(factor(test_pred_res_f$Is_Response), 
                factor(test_pred_res_f$Is_Response_actual),
                mode = "everything")









################################################################################
########testing####out sample ################################
#################################################################################

getwd()

library(readr)
test_out_sample <- read_delim("test.csv", "~", escape_double = TRUE, 
                   trim_ws = TRUE)
View(test_out_sample)




test_out_sample_tokens = test_out_sample$Description %>% 
  prep_fun %>% 
  tok_fun


it_test_out_sample = itoken(test_out_sample_tokens, 
                 progressbar = FALSE)

dtm_test_out_sample_desc = create_dtm(it_test_out_sample, vectorizer)

##################out sample test Browser###################################################
###########################

test_out_sample_tokens2 = test_out_sample$Browser_Used %>% 
  prep_fun %>% 
  tok_fun

################
it_test_out_sample2 = itoken(test_out_sample_tokens2, 
                  progressbar = FALSE)
dtm_test_out_sample_Browser = create_dtm(it_test_out_sample2, vectorizer1)
###############################
##############################################################################



########################out sample test Device#############################################
###########################

test_out_sample_tokens3 = test_out_sample$Device_Used %>% 
  prep_fun %>% 
  tok_fun

################
it_test_out_sample3 = itoken(test_out_sample_tokens3, 
                  progressbar = FALSE)

dtm_test_out_sample_device = create_dtm(it_test_out_sample3, vectorizer2)


########################Device#############################################
###########################



# combine out sample test - Description, Device and Browser
dtm_test_out_sample_all = cbind(dtm_test_out_sample_desc, dtm_test_out_sample_Browser, dtm_test_out_sample_device)
dim(dtm_test_out_sample_all)

xgmat_Pred_out_sample <- xgb.DMatrix(dtm_test_out_sample_all)

#prediction on Hold out sample
preds_xgb <- predict(bst_model, xgmat_Pred_out_sample)

head(preds_xgb)


PRED_test_out_sample = as.data.frame(matrix(preds_xgb, ncol = 2, byrow=TRUE))


library(stringr)
# Mutate xgb output to deliver hard predictions
PRED_test_out_sample$Description = test_out_sample$Description
PRED_test_out_sample$User_ID = test_out_sample$User_ID
PRED_test_out_sample$max_prob = colnames(PRED_test_out_sample)[apply(PRED_test_out_sample[1:2],1,which.max)]
PRED_test_out_sample$level  <- substr(PRED_test_out_sample$max_prob, 2, nchar(PRED_test_out_sample$max_prob))

#joining with good and bad class
unq_response$level <- as.factor(unq_response$level)
test_pred_res$level <- as.factor(test_pred_res$level)

test_pred_res_f <- left_join(test_pred_res, unq_response,by = 'level' )

#test_pred_res_f <- left_join(x = test_pred_res, y = unq_assign,by = 'lbs' )
PRED_test_out_sample_f <- left_join(PRED_test_out_sample, unq_response,by = 'level' )

colnames(PRED_test_out_sample_f)

final_res <- PRED_test_out_sample_f[,c(4,7)]



write.table(final_res, file = "final_res.txt", sep = '~',row.names = FALSE,quote = FALSE
            )


write.table(final_res, file = "final_res.csv", sep = '~',row.names = FALSE,quote = FALSE
)




