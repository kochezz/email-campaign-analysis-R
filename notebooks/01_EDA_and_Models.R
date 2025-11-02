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
