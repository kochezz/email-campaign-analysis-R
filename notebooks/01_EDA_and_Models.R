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

cat("All libraries loaded successfully!\n\n")

# ============================================================
# EMAIL CAMPAIGN — CLEAN TRAIN/TEST EVALUATION
# ============================================================

set.seed(123)


# ---- Import ----
data <- read.csv("Email Campaign.csv", header = TRUE)

cat("===== DATA IMPORT SUMMARY =====\n")
cat("Rows:", nrow(data), "  Cols:", ncol(data), "\n\n")
cat("First 6 rows:\n"); print(head(data)); cat("\n")
cat("===== STRUCTURE =====\n"); str(data); cat("\n")
cat("===== SUMMARY =====\n"); print(summary(data)); cat("\n")
cat("===== MISSING VALUES =====\n")
cat("Total missing:", sum(is.na(data)), "\n")
print(colSums(is.na(data))); cat("\n")
cat("===== TARGET DISTRIBUTION =====\n")
print(table(data$Success)); print(prop.table(table(data$Success))); cat("\n")

# ---- Data prep ----
data$Success <- as.factor(data$Success)   # ensure "0","1" as strings
data$Gender  <- as.factor(data$Gender)
data$AGE     <- as.factor(data$AGE)
data <- data %>% select(-SN)

cat("Variables after prep:\n"); print(names(data)); cat("\n")

# ---- Train/Test split (stratified) ----
set.seed(123)
idx <- caret::createDataPartition(data$Success, p = 0.8, list = FALSE)
train <- data[idx, ]
test  <- data[-idx, ]

cat("Train size:", nrow(train), " Test size:", nrow(test), "\n\n")

# ============================================================
# Q1: DECISION TREE (evaluate on TEST)
# ============================================================

cat("===== DECISION TREE =====\n")
dt_model <- rpart(Success ~ ., data = train, method = "class",
                  control = rpart.control(cp = 0.01))
print(dt_model); cat("\n")

# Plot tree
rpart.plot(dt_model, main = "Decision Tree for Email Campaign Success",
           extra = 104, box.palette = "GnBu", branch.lty = 3,
           shadow.col = "gray", nn = TRUE)

# Test-set probabilities & ROC/AUC
dt_prob_test <- predict(dt_model, test, type = "prob")[,2]
dt_roc <- roc(response = test$Success, predictor = dt_prob_test,
              levels = c("0","1"), direction = "<")
dt_auc <- auc(dt_roc)
cat("Decision Tree AUC (test):", round(dt_auc, 4), "\n")

# Thresholded predictions at 0.50 (simple, comparable across models)
dt_pred_class <- factor(ifelse(dt_prob_test >= 0.50, "1", "0"), levels = c("0","1"))
dt_cm <- confusionMatrix(dt_pred_class, test$Success, positive = "1")
cat("Confusion Matrix (DT, test, cutoff=0.50):\n"); print(dt_cm); cat("\n")
dt_sens <- unname(dt_cm$byClass["Sensitivity"])
dt_acc  <- unname(dt_cm$overall["Accuracy"])

# ============================================================
# Q2: RANDOM FOREST (evaluate on TEST)
# ============================================================

cat("===== RANDOM FOREST =====\n")
set.seed(123)
rf_model <- randomForest(Success ~ ., data = train,
                         ntree = 500, mtry = 3, importance = TRUE)
print(rf_model); cat("\n")

cat("Variable importance (Gini):\n")
imp <- as.data.frame(importance(rf_model)) |>
  tibble::rownames_to_column("Feature") |>
  arrange(desc(MeanDecreaseGini))
print(imp); cat("\n")
varImpPlot(rf_model, main = "Variable Importance - Random Forest")

# Test-set probabilities & ROC/AUC
rf_prob_test <- predict(rf_model, test, type = "prob")[,2]
rf_roc <- roc(response = test$Success, predictor = rf_prob_test,
              levels = c("0","1"), direction = "<")
rf_auc <- auc(rf_roc)
cat("Random Forest AUC (test):", round(rf_auc, 4), "\n")

# Thresholded predictions at 0.50
rf_pred_class <- factor(ifelse(rf_prob_test >= 0.50, "1", "0"), levels = c("0","1"))
rf_cm <- confusionMatrix(rf_pred_class, test$Success, positive = "1")
cat("Confusion Matrix (RF, test, cutoff=0.50):\n"); print(rf_cm); cat("\n")
rf_sens <- unname(rf_cm$byClass["Sensitivity"])
rf_acc  <- unname(rf_cm$overall["Accuracy"])

# Plot DT vs RF ROC on test
plot(dt_roc, col = "blue", lwd = 2, main = "ROC Curves (Test): DT vs RF")
lines(rf_roc, col = "red", lwd = 2)
legend("bottomright",
       legend = c(paste("Decision Tree (AUC =", round(dt_auc, 4), ")"),
                  paste("Random Forest (AUC =", round(rf_auc, 4), ")")),
       col = c("blue", "red"), lwd = 2)

# ============================================================
# Q3: NEURAL NETWORK (clean one-hot + no leakage + test eval)
# ============================================================

cat("===== NEURAL NETWORK =====\n")
# Copy and convert to numeric for NN
nn_train <- train
nn_test  <- test

# Convert target to numeric 0/1 for NN computation (keep factors for ROC later as needed)
nn_train$Success <- as.numeric(as.character(nn_train$Success))
nn_test$Success  <- as.numeric(as.character(nn_test$Success))

# Numeric gender
nn_train$Gender <- as.numeric(as.character(nn_train$Gender))
nn_test$Gender  <- as.numeric(as.character(nn_test$Gender))

# One-hot for AGE but drop one level to avoid perfect collinearity
nn_train <- nn_train %>%
  mutate(
    AGE_30 = as.numeric(AGE == "<=30"),
    AGE_45 = as.numeric(AGE == "<=45")
  ) %>%
  select(-AGE)

nn_test <- nn_test %>%
  mutate(
    AGE_30 = as.numeric(AGE == "<=30"),
    AGE_45 = as.numeric(AGE == "<=45")
  ) %>%
  select(-AGE)

# Normalize predictors using TRAIN stats only (no leakage)
norm_fit <- sapply(nn_train[-1], function(x) c(min = min(x), max = max(x)))
norm_fn <- function(x, mn, mx) if (mx > mn) (x - mn)/(mx - mn) else x*0

nn_train_norm <- nn_train
for (nm in names(nn_train_norm)[-1]) {
  nn_train_norm[[nm]] <- norm_fn(nn_train[[nm]], norm_fit["min", nm], norm_fit["max", nm])
}

nn_test_norm <- nn_test
for (nm in names(nn_test_norm)[-1]) {
  nn_test_norm[[nm]] <- norm_fn(nn_test[[nm]], norm_fit["min", nm], norm_fit["max", nm])
}

# Build NN
nn_formula <- as.formula(paste("Success ~", paste(names(nn_train_norm)[-1], collapse = " + ")))
set.seed(123)
nn_model <- neuralnet(nn_formula, data = nn_train_norm,
                      hidden = c(5,3), linear.output = FALSE, threshold = 0.01, stepmax = 1e6)
plot(nn_model, rep = "best", main = "Neural Network Architecture")

# Predict on TEST
nn_prob_test <- as.vector(predict(nn_model, nn_test_norm))

# ROC/AUC on TEST (use factor for response; pROC needs levels in c("0","1"))
nn_resp_test_factor <- factor(nn_test$Success, levels = c(0,1), labels = c("0","1"))
nn_roc <- roc(response = nn_resp_test_factor, predictor = nn_prob_test,
              levels = c("0","1"), direction = "<")
nn_auc <- auc(nn_roc)
cat("Neural Network AUC (test):", round(nn_auc, 4), "\n")

# Thresholded predictions at 0.50
nn_pred_class <- factor(ifelse(nn_prob_test >= 0.50, "1", "0"), levels = c("0","1"))
nn_cm <- confusionMatrix(nn_pred_class, nn_resp_test_factor, positive = "1")
cat("Confusion Matrix (NN, test, cutoff=0.50):\n"); print(nn_cm); cat("\n")
nn_sens <- unname(nn_cm$byClass["Sensitivity"])
nn_acc  <- unname(nn_cm$overall["Accuracy"])

# Plot NN ROC
plot(nn_roc, col = "darkgreen", lwd = 2, main = "Neural Network ROC (Test)")
legend("bottomright", legend = paste("Neural Network (AUC =", round(nn_auc,4), ")"),
       col = "darkgreen", lwd = 2)

# ============================================================
# FINAL SUMMARY (TEST-SET METRICS)
# ============================================================

cat("============================================================\n")
cat("FINAL SUMMARY: ALL MODELS (TEST SET)\n")
cat("============================================================\n\n")

comparison <- data.frame(
  Model = c("Decision Tree", "Random Forest", "Neural Network"),
  AUC = c(round(as.numeric(dt_auc), 4),
          round(as.numeric(rf_auc), 4),
          round(as.numeric(nn_auc), 4)),
  Sensitivity = c(round(dt_sens, 4),
                  round(rf_sens, 4),
                  round(nn_sens, 4)),
  Accuracy = c(round(dt_acc, 4),
               round(rf_acc, 4),
               round(nn_acc, 4))
)

print(comparison); cat("\n")

best_idx <- which.max(comparison$AUC)
cat("========================================\n")
cat("BEST MODEL BY AUC (TEST):", comparison$Model[best_idx], "\n")
cat("AUC:", comparison$AUC[best_idx], "\n")
cat("========================================\n\n")

# Plot all ROC curves together (TEST)
plot(dt_roc, col = "blue", lwd = 2, main = "ROC Curves: All Models (Test)")
lines(rf_roc, col = "red", lwd = 2)
lines(nn_roc, col = "darkgreen", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Decision Tree (AUC =", round(dt_auc, 4), ")"),
                  paste("Random Forest (AUC =", round(rf_auc, 4), ")"),
                  paste("Neural Network (AUC =", round(nn_auc, 4), ")")),
       col = c("blue", "red", "darkgreen"),
       lwd = 2, cex = 0.8)

# ============================================================
# OPTIONAL: choose a threshold via Youden's J (on TEST, for illustration)
# NOTE: for unbiased thresholding, pick on validation folds, not the test set.
# ============================================================
# dt_thr <- coords(dt_roc, x = "best", best.method = "youden", ret = "threshold")
# rf_thr <- coords(rf_roc, x = "best", best.method = "youden", ret = "threshold")
# nn_thr <- coords(nn_roc, x = "best", best.method = "youden", ret = "threshold")
# cat("Suggested thresholds (optimistic, chosen on test): DT", round(dt_thr,3),
#     " RF", round(rf_thr,3), " NN", round(nn_thr,3), "\n")



























