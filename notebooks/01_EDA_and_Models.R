# ============================================================================
# EMAIL CAMPAIGN CLASSIFICATION ANALYSIS
# Skin Care Clinic Marketing Campaign Success Prediction
# Author: William C. Phiri
# Date: 02 Nov 2025
# Pillar 1e: - Module 8 - Neural Networks
# ============================================================================

# Clear environment
rm(list = ls())

# ============================================================================
# STEP 1: LOAD REQUIRED LIBRARIES
# ============================================================================

install.packages("rpart")        # For decision trees
install.packages("rpart.plot")   # For plotting decision trees
install.packages("caret")        # For confusion matrix and model evaluation
install.packages("pROC")         # For ROC curve analysis
install.packages("randomForest") # For random forest algorithm
install.packages("neuralnet")    # For neural networks
install.packages("dplyr")        # For data manipulation


# Load libraries
library(rpart)          # Decision tree algorithm
library(rpart.plot)     # Visualization of decision trees
library(caret)          # Model evaluation tools
library(pROC)           # ROC curve and AUC calculation
library(randomForest)   # Random forest algorithm
library(neuralnet)      # Neural network algorithm
library(dplyr)          # Data manipulation

set.seed(123)

## ---- Load & quick QA ----
data <- read.csv("Email Campaign.csv", header = TRUE, stringsAsFactors = FALSE)

cat("Rows:", nrow(data), " Cols:", ncol(data), "\n")
cat("Missing by column:\n"); print(colSums(is.na(data)))

## ---- Prep ----
# Ensure valid factor labels for caret ("No","Yes")
data$Success <- factor(data$Success, levels = c(0,1), labels = c("No","Yes"))
data$Gender  <- factor(data$Gender)
data$AGE     <- factor(data$AGE)

# Drop serial number if present
if ("SN" %in% names(data)) data$SN <- NULL

cat("Target distribution:\n"); print(prop.table(table(data$Success)))

## ---- Train/Test split (stratified) ----
idx  <- caret::createDataPartition(data$Success, p = 0.8, list = FALSE)
train <- data[idx, ]
test  <- data[-idx, ]
cat("Train:", nrow(train), " Test:", nrow(test), "\n")

## ---- Helpers ----
# AUC with explicit positive class and ROC object
auc_p <- function(truth, prob_yes) {
  roc_obj <- pROC::roc(response = truth, predictor = prob_yes,
                       levels = c("No","Yes"), direction = "<")
  list(roc = roc_obj, auc = as.numeric(pROC::auc(roc_obj)))
}

# Choose threshold from CV out-of-fold predictions using Youden's J
youden_from_cv <- function(truth, prob_yes) {
  roc_cv <- pROC::roc(truth, prob_yes, levels = c("No","Yes"), direction = "<")
  info <- pROC::coords(roc_cv, x = "best", best.method = "youden",
                       ret = c("threshold","sensitivity","specificity"))
  list(
    thr  = as.numeric(info["threshold"]),
    sens = as.numeric(info["sensitivity"]),
    spec = as.numeric(info["specificity"]),
    roc  = roc_cv
  )
}

## =========================
## Decision Tree (rpart)
## =========================
dt_fit <- rpart(Success ~ ., data = train, method = "class",
                control = rpart.control(cp = 0.01))
rpart.plot(dt_fit, main = "Decision Tree", extra = 104, box.palette = "GnBu")

dt_prob_test <- predict(dt_fit, newdata = test, type = "prob")[,"Yes"]
dt_auc_res   <- auc_p(test$Success, dt_prob_test)
cat("Decision Tree Test AUC:", round(dt_auc_res$auc, 4), "\n")

dt_pred_class <- factor(ifelse(dt_prob_test >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_dt <- caret::confusionMatrix(dt_pred_class, test$Success, positive = "Yes")
print(cm_dt)

## =========================
## Random Forest
## =========================
set.seed(123)
rf_fit <- randomForest(Success ~ ., data = train, ntree = 500, mtry = 3, importance = TRUE)
varImpPlot(rf_fit, main = "Random Forest Variable Importance")

rf_prob_test <- predict(rf_fit, newdata = test, type = "prob")[,"Yes"]
rf_auc_res   <- auc_p(test$Success, rf_prob_test)
cat("Random Forest Test AUC:", round(rf_auc_res$auc, 4), "\n")

rf_pred_class <- factor(ifelse(rf_prob_test >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_rf <- caret::confusionMatrix(rf_pred_class, test$Success, positive = "Yes")
print(cm_rf)

## =========================
## Neural Network (caret::avNNet)
## - CV with centering/scaling inside folds (no leakage)
## - Small nets + L2 decay to reduce overfitting
## - Threshold selected from CV predictions (not test)
## =========================
# avNNet is a caret model; no extra package needed beyond caret
set.seed(123)
ctrl <- trainControl(
  method = "repeatedcv", number = 5, repeats = 3,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final"   # keep out-of-fold predictions for thresholding
)

grid <- expand.grid(
  size  = c(1, 2, 3),           # small networks to avoid overfitting
  decay = c(0.001, 0.01, 0.1),  # L2 regularization
  bag   = FALSE                 # set TRUE to enable model averaging
)

nn_cv <- train(
  Success ~ ., data = train,
  method = "avNNet",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid,
  preProcess = c("center","scale"),
  linout = FALSE,
  trace = FALSE
)
print(nn_cv$bestTune)

# Threshold from CV (no test peeking)
cv_pred <- nn_cv$pred
best    <- nn_cv$bestTune
cv_pred_best <- subset(cv_pred, size == best$size & decay == best$decay)

thr_cv <- youden_from_cv(truth = cv_pred_best$obs, prob_yes = cv_pred_best$Yes)
cat("NN (CV) best tune: size =", best$size, " decay =", best$decay, "\n")
cat("CV-derived threshold (Youden):", round(thr_cv$thr, 3),
    "| Sens:", round(thr_cv$sens, 3),
    "| Spec:", round(thr_cv$spec, 3), "\n")

# Test-set evaluation for NN
nn_prob_test <- predict(nn_cv, newdata = test, type = "prob")[,"Yes"]
nn_auc_res   <- auc_p(test$Success, nn_prob_test)
cat("Neural Net Test AUC:", round(nn_auc_res$auc, 4), "\n")

nn_pred_class <- factor(ifelse(nn_prob_test >= thr_cv$thr, "Yes", "No"), levels = c("No","Yes"))
cm_nn <- caret::confusionMatrix(nn_pred_class, test$Success, positive = "Yes")
print(cm_nn)

## =========================
## ROC plot (test set) – all models
## =========================
dt_roc <- dt_auc_res$roc
rf_roc <- rf_auc_res$roc
nn_roc <- nn_auc_res$roc

plot(dt_roc, col = "steelblue", lwd = 2, main = "ROC Curves (Test Set)")
lines(rf_roc, col = "firebrick", lwd = 2)
lines(nn_roc, col = "darkgreen", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray50")
legend("bottomright",
       legend = c(
         paste0("Decision Tree (AUC=", round(pROC::auc(dt_roc), 3),")"),
         paste0("Random Forest (AUC=", round(pROC::auc(rf_roc), 3),")"),
         paste0("Neural Net (AUC=", round(pROC::auc(nn_roc), 3),")")
       ),
       col = c("steelblue","firebrick","darkgreen"), lwd = 2, cex = 0.9)

## =========================
## Final comparison table (test set)
## =========================
comparison <- data.frame(
  Model       = c("Decision Tree","Random Forest","Neural Net (CV-tuned)"),
  AUC         = c(round(dt_auc_res$auc,4), round(rf_auc_res$auc,4), round(nn_auc_res$auc,4)),
  Sensitivity = round(c(cm_dt$byClass["Sensitivity"], cm_rf$byClass["Sensitivity"], cm_nn$byClass["Sensitivity"]), 4),
  Specificity = round(c(cm_dt$byClass["Specificity"], cm_rf$byClass["Specificity"], cm_nn$byClass["Specificity"]), 4),
  Accuracy    = round(c(cm_dt$overall["Accuracy"],   cm_rf$overall["Accuracy"],   cm_nn$overall["Accuracy"]), 4)
)
print(comparison)

best_idx <- which.max(comparison$AUC)
cat("\nBest by AUC:", comparison$Model[best_idx], " — AUC:", comparison$AUC[best_idx], "\n")








