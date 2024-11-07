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
dat[, metric := factor(metric, levels = c("accuracy", "auc", "logloss"), labels = c("Accuracy", "AUC ROC", "Logloss"))]
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

q05_valid = dat[, .(q05_valid = quantile(valid, p=0.05)), by = .(data_id, train_valid_size, seed, classifier, metric)]
q05_test = dat[, .(q05_test = quantile(test, p=0.05)), by = .(data_id, train_valid_size, seed, classifier, metric)]
dat = merge(dat, q05_valid, by = c("data_id", "train_valid_size", "seed", "classifier", "metric"))
dat = merge(dat, q05_test, by = c("data_id", "train_valid_size", "seed", "classifier", "metric"))
dat[, gap_q05_valid := pmax(valid - q05_valid, 0)]
dat[, gap_q05_test := pmax(test - q05_test, 0)]


### valid, test, normalized valid, normalized test and gap to top 5 valid and test
agg = dat[, .(
  mean_valid = mean(valid),
  se_valid = sd(valid) / sqrt(.N),
  mean_test = mean(test),
  se_test = sd(test) / sqrt(.N),
  mean_normalized_valid = mean(normalized_valid),
  se_normalized_valid = sd(normalized_valid) / sqrt(.N),
  mean_normalized_test = mean(normalized_test),
  se_normalized_test = sd(normalized_test) / sqrt(.N),
  mean_gap_q05_valid = mean(gap_q05_valid),
  se_gap_q05_valid = sd(gap_q05_valid) / sqrt(.N),
  mean_gap_q05_test = mean(gap_q05_test),
  se_gap_q05_test = sd(gap_q05_test) / sqrt(.N)
), by = .(iteration, train_valid_size, resampling, reshuffle, method, metric)]

agg[, n_model_fits := iteration]
agg[resampling == "cv_5_1", n_model_fits := n_model_fits * 5]
agg[resampling == "repeatedholdout_02_5", n_model_fits := n_model_fits * 5]
agg[resampling == "cv_5_5", n_model_fits := n_model_fits * 5 * 5]

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/valid_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_normalized_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Normalized Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/normalized_valid_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_gap_q05_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_gap_q05_valid - se_gap_q05_valid, ymax = mean_gap_q05_valid + se_gap_q05_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Gap to top 5% Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/gap_05_valid_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/test_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_normalized_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Normalized Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/normalized_test_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_gap_q05_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_gap_q05_test - se_gap_q05_test, ymax = mean_gap_q05_test + se_gap_q05_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Gap to top 5% Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/gap_05_test_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg[iteration == 500]) + 
  geom_point(aes(x = n_model_fits, y = mean_normalized_test, colour = resampling, shape = reshuffle), size = 3) +
  geom_errorbar(aes(x = n_model_fits, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, colour = resampling, linetype = reshuffle), width = 0.5) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  xlab("No. Final Model Fits") +
  ylab("Mean Normalized Test Performance") +
  labs(color = "Resampling", shape = "Reshuffling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/pareto_normalized_test_n_model_fits.pdf", width = 9, height = 6)



ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Validation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_valid_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_normalized_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Normalized\nValidation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_normalized_valid_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_gap_q05_valid, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_gap_q05_valid - se_gap_q05_valid, ymax = mean_gap_q05_valid + se_gap_q05_valid, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Gap to top 5%\nValidation Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_gap_05_valid_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_test_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_normalized_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Normalized\nTest Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_normalized_test_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_gap_q05_test, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymin = mean_gap_q05_test - se_gap_q05_test, ymax = mean_gap_q05_test + se_gap_q05_test, fill = resampling, linetype = reshuffle), alpha = 0.1, colour = NA) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Gap to top 5%\nTest Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_gap_05_test_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC" & iteration == 500]) + 
  geom_point(aes(x = n_model_fits, y = mean_normalized_test, colour = resampling, shape = reshuffle), size = 3) +
  geom_errorbar(aes(x = n_model_fits, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, colour = resampling, linetype = reshuffle), width = 0.5) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  xlab("No. Final Model Fits") +
  ylab("Mean Normalized\nTest Performance") +
  labs(color = "Resampling", shape = "Reshuffling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_pareto_normalized_test_n_model_fits.pdf", width = 9, height = 3)



### improvement and relative performance
agg = dat[, .(
  mean_log_rel_test = mean(log(rel_test)),
  se_log_rel_test = sd(log(rel_test)) / sqrt(.N),
  mean_improvement_rel_test_baseline = mean(improvement_rel_test_baseline),
  se_improvement_rel_test_baseline = sd(improvement_rel_test_baseline) / sqrt(.N)
), by = .(iteration, train_valid_size, resampling, reshuffle, method, metric)]

agg[, n_model_fits := iteration]
agg[resampling == "cv_5_1", n_model_fits := n_model_fits * 5]
agg[resampling == "repeatedholdout_02_5", n_model_fits := n_model_fits * 5]
agg[resampling == "cv_5_5", n_model_fits := n_model_fits * 5 * 5]

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = mean_improvement_rel_test_baseline, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Test Improvement") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/imp_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg) +
  geom_step(aes(x = n_model_fits, y = exp(mean_log_rel_test), colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  geom_hline(yintercept = 1, linetype = 3) +
  facet_wrap(~ metric + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Relative Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/rel_n_model_fits.pdf", width = 9, height = 6)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = mean_improvement_rel_test_baseline, colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Test Improvement") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_imp_n_model_fits.pdf", width = 9, height = 3)

ggplot(data = agg[metric == "AUC ROC"]) +
  geom_step(aes(x = n_model_fits, y = exp(mean_log_rel_test), colour = resampling, linetype = reshuffle)) +
  geom_stepribbon(aes(x = n_model_fits, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = resampling, linetype = reshuffle), colour = NA, alpha = 0.1) +
  geom_hline(yintercept = 1, linetype = 3) +
  facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
  scale_x_log10() +
  scale_color +
  scale_fill +
  xlab("No. Model Fits") +
  ylab("Mean Relative Test Performance") +
  labs(color = "Resampling", fill = "Resampling", linetype = "Reshuffling") +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("plots/auc_rel_n_model_fits.pdf", width = 9, height = 3)

