library(data.table)
library(ggplot2)
library(ggthemes)
library(pammtools)
library(mlr3misc)

scale_fill_colorblind7 = function(.ColorList = 1L:8L, ...) {
  scale_fill_discrete(..., type = colorblind_pal()(8)[.ColorList])
}

scale_colour_colorblind7 = function(.ColorList = 1L:8L, ...) {
  scale_colour_discrete(..., type = colorblind_pal()(8)[.ColorList])
}

scale_fill = scale_fill_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV"))
scale_color = scale_colour_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV"))

dat_holdout = fread("csvs/results_holdout_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
dat_cv = fread("csvs/results_cv_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
dat_cv = dat_cv[method %in% c("cv_5_1_False_post_naive", "cv_5_1_True_post_naive")]
dat_cv_repeated = fread("csvs/results_cv_repeated_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
dat_cv_repeated = dat_cv_repeated[method %in% c("cv_5_5_False_post_naive", "cv_5_5_True_post_naive")]
dat_repeatedholdout = fread("csvs/results_repeatedholdout_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
dat_repeatedholdout = dat_repeatedholdout[method %in% c(
  "cv_5_5_False_post_naive_simulate_repeatedholdout_5", "cv_5_5_True_post_naive_simulate_repeatedholdout_5")]
dat_holdout[, resampling_orig := resampling]
dat_cv[, resampling_orig := resampling]
dat_cv_repeated[, resampling_orig := resampling]
dat_repeatedholdout[, resampling_orig := resampling]
dat_repeatedholdout[method == "cv_5_5_False_post_naive_simulate_repeatedholdout_5", resampling := "repeatedholdout_02_5_False"]
dat_repeatedholdout[method == "cv_5_5_False_post_naive_simulate_repeatedholdout_5", method := "repeatedholdout_02_5_False_post_naive"]
dat_repeatedholdout[method == "cv_5_5_True_post_naive_simulate_repeatedholdout_5", resampling := "repeatedholdout_02_5_True"]
dat_repeatedholdout[method == "cv_5_5_True_post_naive_simulate_repeatedholdout_5", method := "repeatedholdout_02_5_True_post_naive"]
dat = rbind(dat_holdout, dat_cv, dat_cv_repeated, dat_repeatedholdout, fill = TRUE)
rm(dat_holdout, dat_cv, dat_cv_repeated, dat_repeatedholdout)

dat = dat[metric != "balanced_accuracy" & optimizer == "random"]
dat = dat[grepl("post_naive", method)]
dat[, seed := as.factor(seed)]
dat[, classifier := factor(classifier, levels = c("logreg", "logreg_default", "funnel_mlp", "funnel_mlp_default", "xgboost", "xgboost_default", "catboost", "catboost_default"))]
dat[, data_id := as.factor(data_id)]
dat[, train_valid_size := as.factor(train_valid_size)]
dat[, reshuffle := grepl("_True", resampling_orig)]
dat[, resampling_orig := gsub("_True|_False", "", resampling_orig)]
dat[, resampling := gsub("_True|_False", "", resampling)]
dat[, metric := factor(metric, levels = c("accuracy", "auc", "logloss"), labels = c("Accuracy", "ROC AUC", "Logloss"))]
dat[, method := as.factor(method)]
dat[, resampling_orig := as.factor(resampling_orig)]
dat[, resampling := as.factor(resampling)]
dat[, reshuffle := as.factor(reshuffle)]
dat = dat[, c("valid", "test", "data_id", "seed", "classifier", "metric", "train_valid_size", "iteration", "resampling_orig", "resampling", "reshuffle", "method")]

defaults = dat[classifier %in% c("logreg_default", "funnel_mlp_default", "xgboost_default", "catboost_default")]
defaults[, default_test := test]
defaults[, classifier := as.factor(gsub("_default", "", classifier))]
defaults = defaults[, c("default_test", "data_id", "seed", "classifier", "metric", "train_valid_size", "resampling_orig", "reshuffle")]

dat = merge(dat, unique((defaults)))
rm(defaults)

dat[, rel_test := test / default_test]
dat[metric == "Logloss", rel_test := default_test / test]

dat[, baseline := FALSE]
dat[method == "cv_5_1_False_post_naive", baseline := TRUE]
baseline = dat[baseline == TRUE, ]
baseline[, test_baseline := test]
baseline = baseline[, c("iteration", "test_baseline", "data_id", "seed", "classifier", "metric", "train_valid_size")]

dat = merge(dat, baseline, by = c("iteration", "data_id", "seed", "classifier", "metric", "train_valid_size"))
dat[, improvement_rel_test_baseline := (test_baseline - test) * 100]

max_min_valid = dat[, .(max_valid = max(valid), min_valid = min(valid)), by = .(data_id, train_valid_size, seed, classifier, metric)]
max_min_test = dat[, .(max_test = max(test), min_test = min(test)), by = .(data_id, train_valid_size, seed, classifier, metric)]
dat = merge(dat, max_min_valid, by = c("data_id", "train_valid_size", "seed", "classifier", "metric"))
dat = merge(dat, max_min_test, by = c("data_id", "train_valid_size", "seed", "classifier", "metric"))
normalize_value = function(value, max_value, min_value) {
  normalized_test = (value - min_value) / (max_value - min_value)
  normalized_test[max_value == min_value] = 0
  normalized_test
}
dat[, normalized_valid := normalize_value(valid, max_valid, min_valid), by = .(data_id, train_valid_size, seed, classifier, metric, resampling)]
dat[, normalized_test := normalize_value(test, max_test, min_test), by = .(data_id, train_valid_size, seed, classifier, metric, resampling)]

### valid, test, normalized valid and normalized test
agg = dat[, .(
  mean_valid = mean(valid),
  se_valid = sd(valid) / sqrt(.N),
  mean_test = mean(test),
  se_test = sd(test) / sqrt(.N),
  mean_normalized_valid = mean(normalized_valid),
  se_normalized_valid = sd(normalized_valid) / sqrt(.N),
  mean_normalized_test = mean(normalized_test),
  se_normalized_test = sd(normalized_test) / sqrt(.N)
), by = .(iteration, train_valid_size, resampling, reshuffle, method, metric)]

ggplot(data = agg) +
  geom_step(aes(x = iteration, y = mean_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/valid.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = iteration, y = mean_normalized_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Normalized Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/normalized_valid.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = iteration, y = mean_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/test.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = iteration, y = mean_normalized_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Normalized Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/normalized_test.pdf", width = 9, height = 6)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_valid.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_normalized_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Normalized\nValidation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_normalized_valid.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_test.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_normalized_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Normalized\nTest Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_normalized_test.pdf", width = 9, height = 3)



### valid, test classifier level
agg = dat[, .(
  mean_valid = mean(valid),
  se_valid = sd(valid) / sqrt(.N),
  mean_test = mean(test),
  se_test = sd(test) / sqrt(.N),
  mean_normalized_valid = mean(normalized_valid),
  se_normalized_valid = sd(normalized_valid) / sqrt(.N),
  mean_normalized_test = mean(normalized_test),
  se_normalized_test = sd(normalized_test) / sqrt(.N)
), by = .(iteration, train_valid_size, classifier, resampling, reshuffle, method, metric)]

classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
  classifierxid = classifierxids[[classifierx]]
  ggplot(data = agg[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = mean_valid, colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Validation Performance") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_valid.pdf"), width = 9, height = 6)

  ggplot(data = agg[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = mean_normalized_valid, colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Normalized Validation Performance") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_normalized_valid.pdf"), width = 9, height = 6)

  ggplot(data = agg[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = mean_test, colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Test Performance") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_test.pdf"), width = 9, height = 6)

  ggplot(data = agg[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = mean_normalized_test, colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Normalized Test Performance") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_normalized_test.pdf"), width = 9, height = 6)
}



### valid, test raw
agg = dat[, .(
  mean_valid = mean(valid),
  se_valid = sd(valid) / sqrt(.N),
  mean_test = mean(test),
  se_test = sd(test) / sqrt(.N),
  mean_normalized_valid = mean(normalized_valid),
  se_normalized_valid = sd(normalized_valid) / sqrt(.N),
  mean_normalized_test = mean(normalized_test),
  se_normalized_test = sd(normalized_test) / sqrt(.N)
), by = .(iteration, data_id, train_valid_size, classifier, resampling, reshuffle, method, metric)]

metricxids = list("accuracy" = "Accuracy", "auc" = "ROC AUC", "logloss" = "Logloss")
classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
  for (metricx in c("accuracy", "auc", "logloss")) {
    classifierxid = classifierxids[[classifierx]]
    metricxid = metricxids[[metricx]]
    subpath = paste0(classifierx, "_", metricx)
    dir.create(file.path("plots/", subpath), showWarnings = FALSE)

    ggplot(data = agg[classifier == classifierx & metric == metricxid]) +
      geom_step(aes(x = iteration, y = mean_valid, colour = resampling, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ data_id + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
      scale_colour_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
      scale_fill_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
      xlab("No. HPC Evaluations") +
      ylab("Mean Validation Performance") +
      labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("subpath", subpath, "plots/subpath/valid.pdf"), width = 9, height = 15)

    ggplot(data = agg[classifier == classifierx & metric == metricxid]) +
      geom_step(aes(x = iteration, y = mean_test, colour = resampling, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ data_id + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
      scale_colour_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
      scale_fill_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
      xlab("No. HPC Evaluations") +
      ylab("Mean Test Performance") +
      labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("subpath", subpath, "plots/subpath/test.pdf"), width = 9, height = 15)
  }
}



### improvement and relative performance
agg = dat[, .(
  mean_log_rel_test = mean(log(rel_test)),
  se_log_rel_test = sd(log(rel_test)) / sqrt(.N),
  mean_improvement_rel_test_baseline = mean(improvement_rel_test_baseline),
  se_improvement_rel_test_baseline = sd(improvement_rel_test_baseline) / sqrt(.N)
), by = .(iteration, train_valid_size, resampling, reshuffle, method, metric)]

ggplot(data = agg) +
  geom_step(aes(x = iteration, y = mean_improvement_rel_test_baseline, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  facet_wrap(~ metric + train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Test Improvement") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/imp.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = iteration, y = exp(mean_log_rel_test), colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  geom_hline(yintercept = 1, linetype = 3) +
  facet_wrap(~ metric + train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Relative Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/rel.pdf", width = 9, height = 6)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_improvement_rel_test_baseline, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  facet_wrap(~ train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Test Improvement") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_imp.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = exp(mean_log_rel_test), colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  geom_hline(yintercept = 1, linetype = 3) +
  facet_wrap(~ train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Relative Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_rel.pdf", width = 9, height = 3)



### improvement and relative performance classifier level
agg = dat[, .(
  mean_log_rel_test = mean(log(rel_test)),
  se_log_rel_test = sd(log(rel_test)) / sqrt(.N),
  mean_improvement_rel_test_baseline = mean(improvement_rel_test_baseline),
  se_improvement_rel_test_baseline = sd(improvement_rel_test_baseline) / sqrt(.N)
), by = .(iteration, classifier, train_valid_size, resampling, reshuffle, method, metric)]

classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
  classifierxid = classifierxids[[classifierx]]
  ggplot(data = agg[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = mean_improvement_rel_test_baseline, colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Test Improvement") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_imp.pdf"), width = 9, height = 6)

  ggplot(data = agg[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = exp(mean_log_rel_test), colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
    geom_hline(yintercept = 1, linetype = 3) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Relative Test Performance") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_rel.pdf"), width = 9, height = 6)
}



### ranks
dat[, rank := frank(test, ties.method = "average"), by = .(iteration, data_id, seed, classifier, metric, train_valid_size)]
ranks = dat[, .(mean_rank = mean(rank), se_rank = sd(rank) / sqrt(.N)), by = .(iteration, metric, resampling, train_valid_size, reshuffle, method)]
ggplot(data = ranks) +
  geom_step(aes(x = iteration, y = mean_rank, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_rank - se_rank, ymax = mean_rank + se_rank, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  facet_wrap(~ metric + train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Rank (Test Performance)") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/rel_ranks.pdf", width = 9, height = 6)

ggplot(data = ranks[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_rank, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_rank - se_rank, ymax = mean_rank + se_rank, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  facet_wrap(~ train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_color +
  scale_fill +
  xlab("No. HPC Evaluations") +
  ylab("Mean Rank (Test Performance)") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_rel_ranks.pdf", width = 9, height = 3)



### ranks classifier level
ranks = dat[, .(mean_rank = mean(rank), se_rank = sd(rank) / sqrt(.N)), by = .(iteration, classifier, metric, resampling, train_valid_size, reshuffle, method)]
classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
  classifierxid = classifierxids[[classifierx]]
  ggplot(data = ranks[classifier == classifierx]) +
    geom_step(aes(x = iteration, y = mean_rank, colour = resampling, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_rank - se_rank, ymax = mean_rank + se_rank, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
    facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Rank (Test Performance)") +
    labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("classifierx", classifierx, "plots/classifierx_rel_ranks.pdf"), width = 9, height = 6)
}



# example plot
tmp = dat[classifier == "xgboost" & data_id == "41147" & metric == "ROC AUC"]
agg = tmp[, .(
  mean_valid = mean(valid),
  se_valid = sd(valid) / sqrt(.N),
  mean_test = mean(test),
  se_test = sd(test) / sqrt(.N),
  mean_normalized_valid = mean(normalized_valid),
  se_normalized_valid = sd(normalized_valid) / sqrt(.N),
  mean_normalized_test = mean(normalized_test),
  se_normalized_test = sd(normalized_test) / sqrt(.N)
), by = .(iteration, data_id, train_valid_size, classifier, resampling, reshuffle, method, metric)]

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_colour_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
  scale_fill_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
  xlab("No. HPC Evaluations") +
  ylab("Mean Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/xgboost_41147_valid.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "ROC AUC"]) +
  geom_step(aes(x = iteration, y = mean_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_continuous(breaks = c(1, 100, 200, 300, 400, 500)) +
  scale_colour_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
  scale_fill_colorblind7(breaks = c("holdout_02", "cv_5_1", "repeatedholdout_02_5", "cv_5_5"), labels = c("Holdout", "5-fold CV", "5-fold Holdout", "5x 5-fold CV")) +
  xlab("No. HPC Evaluations") +
  ylab("Mean Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/xgboost_41147_test.pdf", width = 9, height = 3)



#library(scmamp)
#
#agg = dat[iteration == 500, .(
#  mean_valid = mean(valid),
#  se_valid = sd(valid) / sqrt(.N),
#  mean_test = mean(test),
#  se_test = sd(test) / sqrt(.N)
#), by = .(iteration, data_id, classifier, metric, resampling, train_valid_size, reshuffle, method)]
#agg[, task := paste0(classifier, "_", data_id, "_", train_valid_size)]
#
#tmp = as.matrix(dcast(agg[metric == "accuracy"], task ~ method, value.var = "mean_test")[, -1L])
#friedmanTest(tmp)
#plotCD(tmp, cex = 1.2)
#
#tmp = as.matrix(dcast(agg[metric == "auc"], task ~ method, value.var = "mean_test")[, -1L])
#friedmanTest(tmp)
#plotCD(tmp, cex = 1.2)
#
#tmp = as.matrix(dcast(agg[metric == "logloss"], task ~ method, value.var = "mean_test")[, -1L])
#friedmanTest(tmp)
#plotCD(tmp, cex = 1.2)
#
#make_paircomp_row = function(x) {
#  k = length(x)
#  #npcs = choose(k, 2)
#  pc = matrix(NA, nrow = k, ncol = k)
#  rownames(pc) = colnames(pc) = names(x)
#  for (i in seq_len(k)) {
#    for (j in seq_len(k)) {
#      if (j > i) {
#        pc[i, j] = if (x[i] == x[j]) 0 else if (x[i] > x[j]) 1 else -1
#      }
#    }
#  }
#  pc[upper.tri(pc)]
#}
#
#agg = dat[iteration == 500, .(
#  mean_valid = mean(valid),
#  se_valid = sd(valid) / sqrt(.N),
#  mean_test = mean(test),
#  se_test = sd(test) / sqrt(.N)
#), by = .(iteration, data_id, classifier, metric, resampling, train_valid_size, reshuffle, method)]
#agg[, task := paste0(classifier, "_", data_id, "_", train_valid_size, "_", metric)]
#
#library(psychotree)
#library(ggparty)
#
#source("attic/R/amlb_tasks.R")
#selected[, data_id := as.factor(data_id)]
#selected[, n_features := NumberOfFeatures]
#selected[, class_ratio := MinorityClassSize / MajorityClassSize]
#selected[, miss_ratio := NumberOfInstancesWithMissingValues / NumberOfInstances]
#selected[, cat_ratio := NumberOfSymbolicFeatures / NumberOfFeatures]
#selected = selected[, c("data_id", "n_features", "class_ratio", "miss_ratio", "cat_ratio")]
#
#tmp = dcast(agg, task ~ method, value.var = "mean_test")
#colnames(tmp) = c("task", "cv5", "cv5s", "5cv5", "5cv5s", "h", "hs", "5h", "5hs")
#tmp = tmp[, c("task", "5cv5", "5cv5s", "cv5", "cv5s", "5h", "5hs", "h", "hs")]
#paircomp_cols = apply(tmp[, -1L], MARGIN = 1L, FUN = make_paircomp_row)
#paircomp_dat = paircomp(t(paircomp_cols), labels = colnames(tmp[, -1L]), mscale = c(-1, 0, 1))
#paircomp_cov = agg[method == "holdout_02_False_post_naive"]
#paircomp_cov = merge(paircomp_cov, selected)
#paircomp = as.data.frame(paircomp_cov[match(tmp$task, task), ][, c("classifier", "n_features", "class_ratio", "miss_ratio", "cat_ratio", "metric", "train_valid_size")])
#paircomp$classifier = as.factor(paircomp$classifier)
#paircomp$metric = as.factor(paircomp$metric)
#paircomp$train_valid_size = as.integer(as.character(paircomp$train_valid_size))
#paircomp = cbind(paircomp, data.frame(response = paircomp_dat))
#
#model = bttree(response ~ ., data = paircomp, maxdepth = 4, alpha = 0.1, ref = "cv5", vcov = "info")
#pdf("plots/bttree.pdf", width = 25, height = 6)
#plot(model, abbreviate = FALSE)
#dev.off()
#
#paircomp_auc = paircomp[paircomp$metric == "auc", ]
#model = bttree(response ~ classifier + n_features + class_ratio + miss_ratio + cat_ratio + train_valid_size, data = paircomp_auc, maxdepth = 4, alpha = 0.1, ref = "cv5", vcov = "info")
#pdf("plots/auc_bttree.pdf", width = 25, height = 6)
#plot(model, abbreviate = FALSE)
#dev.off()
