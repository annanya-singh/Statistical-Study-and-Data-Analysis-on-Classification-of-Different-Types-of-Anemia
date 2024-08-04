# Statistical-Study-and-Data-Analysis-on-Classification-of-Different-Types-of-Anemia
This project aims to develop a machine learning model to classify anemia types using Complete Blood Count (CBC) reports. It seeks to identify key hematological factors indicative of various anemia types through data analysis, model development, and feature importance interpretation, aiding accurate diagnosis.

data = read.csv(choose.files())

head(data)
summary(data)
str(data)

# Libraries

library(ggplot2)
library(gridExtra)
library(dplyr)
library(qqplotr)
library(nnet)
library(MASS)
library(caret)
library(reshape2)
library(gridExtra)
library(lmtest)
library(rpart)         
library(randomForest)


# Model Learning

# Define the partition
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE, times = 1)

# Split the data
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Print the number of samples in the training and testing sets
cat("X_train samples:", length(y_train), "\n")
cat("X_test samples:", length(y_test), "\n")


# Convert "Diagnosis" into a factor
data$Diagnosis <- as.factor(data$Diagnosis)

# Split the data into training and test sets
set.seed(123)  # For reproducibility
sample <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[sample, ]
test_data <- data[-sample, ]

# Fit a multinomial logistic regression model
model <- multinom(Diagnosis ~ WBC + LYMp + NEUTp + LYMn + NEUTn + RBC + HGB + HCT + MCV + MCH + MCHC + PLT + PDW + PCT, data = train_data)
model

# Check the summary of the model
summary(model)

# Perform stepwise selection using AIC as the criterion
stepwise_model <- stepAIC(model, direction = "both")

stepwise_model
summary(stepwise_model)

# Make predictions on the test set
predicted <- predict(model, newdata = test_data)
predicted <- predict(stepwise_model, newdata = test_data)

# Evaluate the model
confusion_matrix <- table(predicted, test_data$Diagnosis)
print(confusion_matrix)


X <- data[, !(names(data) %in% "Diagnosis")]
y <- data$Diagnosis


# Likelihood Ratio Test
lr_test <- lrtest(stepwise_model)

# Print test results
print(lr_test)


# Train the Random Forest classification model
model2 <- randomForest(Diagnosis ~ ., data = data, importance = TRUE)

# Get feature importances
importance_scores <- importance(model2)

# Convert to a data frame for easier plotting
importance_df <- data.frame(Feature = rownames(importance_scores), Importance = importance_scores[, "MeanDecreaseGini"])

# Plot feature importances
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Feature") +
  ylab("Importance") +
  ggtitle("Feature Importance using Random Forest")


#MODEL BUILDING

# Ensure y_pred and y_true have the same levels
ensure_levels <- function(y_pred, y_true) {
  levels(y_pred) <- levels(y_true)
  return(y_pred)
}

# Function to calculate balanced accuracy
balanced_accuracy <- function(y_true, y_pred) {
  y_pred <- ensure_levels(y_pred, y_true)
  cm <- confusionMatrix(y_pred, y_true)
  return(mean(cm$byClass[,"Balanced Accuracy"]))
}

# Initialize models
SEED <- 42
models <- list(
  dt = rpart,
  rf = randomForest
)

# Loop through models
for (name in names(models)) {
  cat(paste0("* ", name, " | "))
  
  # Fit model
  if (name == "dt") {
    model <- models[[name]](y_train ~ ., data = X_train)
    y_pred_train <- predict(model, X_train, type = "class")
    y_pred_test <- predict(model, X_test, type = "class")
  } else if (name == "rf") {
    model <- models[[name]](X_train, y_train, ntree = 100)
    y_pred_train <- predict(model, X_train)
    y_pred_test <- predict(model, X_test)
  }
  
  # Calculate balanced accuracy
  acc_train <- balanced_accuracy(y_train, y_pred_train)
  acc_test <- balanced_accuracy(y_test, y_pred_test)
  cat(sprintf("Acc Train: %.4f | Acc Test: %.4f\n", acc_train, acc_test))
}

# Plots

# Select numeric columns
features <- data %>% select_if(is.numeric) %>% names()

# Create color palette
colors <- scales::hue_pal()(length(features))

# Create individual plots and store in a list
plots <- list()
for (i in seq_along(features)) {
  p <- ggplot(data, aes_string(x = features[i])) +
    geom_histogram(aes(y = ..density..), fill = NA, color = colors[i], bins = 30) +
    geom_density(fill = colors[i], alpha = 0.4) +
    labs(title = features[i]) +
    theme_light() +
    theme(plot.title = element_text(size = 10, face = "bold"),
          axis.title.x = element_blank())
  plots[[i]] <- p
}

# Arrange plots in a 7x2 grid
grid.arrange(grobs = plots, ncol = 2, nrow = 7)


# Create individual Q-Q plots and store in a list
plots <- list()
for (i in seq_along(features)) {
  p <- ggplot(data, aes(sample = !!sym(features[i]))) +
    stat_qq_line() +
    stat_qq_point() +
    labs(title = features[i]) +
    theme_light() +
    theme(plot.title = element_text(size = 10, face = "bold"),
          axis.title.x = element_blank(),
          axis.title.y = element_blank())
  plots[[i]] <- p
}

# Arrange plots in a 7x2 grid
grid.arrange(grobs = plots, ncol = 4, nrow = 4)


# Create the count plot with horizontal bars and adjusted text
plot <- ggplot(data, aes(x = Diagnosis)) +
  geom_bar(fill = "lightblue", color = "black") +
  geom_text(stat = "count", aes(label = ..count..), hjust = -0.1, size = 3, fontface = "bold") +
  labs(x = "", y = "Count", title = "Diagnosis") +
  theme_light() +
  theme(
    plot.title = element_text(size = 14, face = "bold", color = "darkblue"),
    axis.text.y = element_text(size = 10, face = "bold", color = "black")
  ) +
  coord_flip()  # Flip the coordinates to make the bar plot horizontal


# Show the plot
print(plot)

# Select only numeric columns
numeric_data <- data %>% select_if(is.numeric)

# Calculate the correlation matrix using the Spearman method
corr_matrix <- cor(numeric_data, method = "spearman")

# Convert the correlation matrix to long format
corr_matrix_long <- melt(corr_matrix)

# Create the heatmap
plot <- ggplot(corr_matrix_long, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "black") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation") +
  geom_text(aes(label = round(value, 2)), size = 3, fontface = "bold") +
  theme_minimal() +
  labs(x = "", y = "", title = "Correlation Matrix", color = "black", 
       title.fontface = "bold", fill = "Correlation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Show the plot
print(plot)


# Convert y_train to a data frame for ggplot
df_y_train <- data.frame(Diagnosis = y_train)

# Create the plot
ggplot(data = df_y_train, aes(x = Diagnosis)) +
  geom_bar(aes(y = after_stat(count)), fill = "skyblue", color = "black", width = 0.7) +
  geom_text(stat = 'count', aes(label = after_stat(count)), hjust = -0.1, size = 3, fontface = "bold", color = "black") +
  labs(title = "Diagnosis Train", y = "", x = "") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", color = "darkblue"),
    axis.text.y = element_text(size = 10, face = "bold", color = "black")
  ) +
  coord_flip()  # Flip the coordinates to match the horizontal bar plot

df_y_test <- data.frame(Diagnosis = y_test)

# Create the plot for y_test
ggplot(data = df_y_test, aes(x = Diagnosis)) +
  geom_bar(aes(y = after_stat(count)), fill = "lightcoral", color = "black", width = 0.7) +
  geom_text(stat = 'count', aes(label = after_stat(count)), hjust = -0.1, size = 3, fontface = "bold", color = "black") +
  labs(title = "Diagnosis Test", y = "", x = "") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", color = "darkblue"),
    axis.text.y = element_text(size = 10, face = "bold", color = "black")
  ) +
  coord_flip()  # Flip the coordinates to match the horizontal bar plot

