library(data.table)
library(ggplot2)
library(ggthemes)
library(pammtools)
library(mlr3misc)

metricx = "auc"
metricxid = "ROC AUC"

scale_fill_colorblind7 = function(.ColorList = 1L:8L, ...) {
  scale_fill_discrete(..., type = colorblind_pal()(8)[.ColorList])
}

scale_colour_colorblind7 = function(.ColorList = 1L:8L, ...) {
  scale_colour_discrete(..., type = colorblind_pal()(8)[.ColorList])
}

scale_fill = scale_fill_colorblind7(breaks = c("random", "hebo", "smac"), labels = c("Random Search", "HEBO", "SMAC3"))
scale_color = scale_colour_colorblind7(breaks = c("random", "hebo", "smac"), labels = c("Random Search", "HEBO", "SMAC3"))

for (valid_type in c("holdout_02", "repeatedholdout_02_5", "cv_5_1", "cv_5_5")) {
  if (valid_type == "holdout_02") {
    dat = fread("csvs/results_holdout_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
  } else if (valid_type == "repeatedholdout_02_5") {
    dat = fread("csvs/results_repeatedholdout_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
    dat = dat[method %in% c("cv_5_5_False_post_naive_simulate_repeatedholdout_5", "cv_5_5_True_post_naive_simulate_repeatedholdout_5", "repeatedholdout_02_5_False_post_naive", "repeatedholdout_02_5_True_post_naive")]
    dat[method == "cv_5_5_False_post_naive_simulate_repeatedholdout_5", resampling := "repeatedholdout_02_5_False"]
    dat[method == "cv_5_5_False_post_naive_simulate_repeatedholdout_5", method := "repeatedholdout_02_5_False_post_naive"]
    dat[method == "cv_5_5_True_post_naive_simulate_repeatedholdout_5", resampling := "repeatedholdout_02_5_True"]
    dat[method == "cv_5_5_True_post_naive_simulate_repeatedholdout_5", method := "repeatedholdout_02_5_True_post_naive"]
  } else if (valid_type == "cv_5_1") {
    dat = fread("csvs/results_cv_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
    dat = dat[method %in% c("cv_5_1_False_post_naive", "cv_5_1_True_post_naive")]
  } else if (valid_type == "cv_5_5") {
    dat = fread("csvs/results_cv_repeated_post_test_retrained.csv")[classifier %nin% c("tabpfn")]
    dat = dat[method %in% c("cv_5_5_False_post_naive", "cv_5_5_True_post_naive")]
  }
  dat[, resampling_orig := resampling]
  dat = dat[grepl("post_naive", method)]
  dat = dat[iteration <= 250 & metric == "auc"]
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
  dat[, optimizer := factor(optimizer, levels = c("random", "hebo", "smac"))]
  dat[optimizer == "hebo", method := paste0(method, "_", "hebo")]
  dat[optimizer == "smac", method := paste0(method, "_", "smac")]
  dat = dat[, c("valid", "test", "data_id", "seed", "classifier", "metric", "train_valid_size", "iteration", "resampling_orig", "resampling", "reshuffle", "method", "optimizer")]

  defaults = dat[classifier %in% c("logreg_default", "funnel_mlp_default", "xgboost_default", "catboost_default")]
  defaults[, default_test := test]
  defaults[, classifier := as.factor(gsub("_default", "", classifier))]
  defaults = defaults[, c("default_test", "data_id", "seed", "classifier", "metric", "train_valid_size", "resampling_orig", "reshuffle")]

  dat = merge(dat, defaults)
  rm(defaults)

  dat[, rel_test := test / default_test]
  dat[metric == "Logloss", rel_test := default_test / test]

  dat[, baseline := FALSE]
  dat[method == paste0(valid_type, "_False_post_naive"), baseline := TRUE]
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
  ), by = .(iteration, train_valid_size, resampling, reshuffle, method, metric, optimizer)]

  ggplot(data = agg) +
    geom_step(aes(x = iteration, y = mean_valid, colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Validation Performance") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_valid.pdf"), width = 9, height = 3)

  ggplot(data = agg) +
    geom_step(aes(x = iteration, y = mean_normalized_valid, colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Normalized\nValidation Performance") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_normalized_valid.pdf"), width = 9, height = 3)

  ggplot(data = agg) +
    geom_step(aes(x = iteration, y = mean_test, colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Test Performance") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_test.pdf"), width = 9, height = 3)

  ggplot(data = agg) +
    geom_step(aes(x = iteration, y = mean_normalized_test, colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
    facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Normalized\nTest Performance") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_normalized_test.pdf"), width = 9, height = 3)



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
  ), by = .(iteration, train_valid_size, classifier, resampling, reshuffle, method, metric, optimizer)]

  classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
  for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
    classifierxid = classifierxids[[classifierx]]
    ggplot(data = agg[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = mean_valid, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Validation Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_valid.pdf")), width = 9, height = 3)

    ggplot(data = agg[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = mean_normalized_valid, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_normalized_valid - se_normalized_valid, ymax = mean_normalized_valid + se_normalized_valid, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Normalized\nValidation Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_normalized_valid.pdf")), width = 9, height = 3)

    ggplot(data = agg[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = mean_test, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Test Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_test.pdf")), width = 9, height = 3)

    ggplot(data = agg[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = mean_normalized_test, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_normalized_test - se_normalized_test, ymax = mean_normalized_test + se_normalized_test, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Normalized\nTest Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_normalized_test.pdf")), width = 9, height = 3)
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
  ), by = .(iteration, data_id, train_valid_size, classifier, resampling, reshuffle, method, metric, optimizer)]

  classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
  for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
    classifierxid = classifierxids[[classifierx]]
    subpath = paste0(classifierx, "_", metricx)

    ggplot(data = agg[classifier == classifierx & metric == metricxid]) +
      geom_step(aes(x = iteration, y = mean_valid, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_valid - se_valid, ymax = mean_valid + se_valid, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ data_id + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line=FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Validation Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("subpath", subpath, gsub("valid_type", valid_type, "plots/subpath/bo_valid_type_valid.pdf")), width = 9, height = 15)

    ggplot(data = agg[classifier == classifierx & metric == metricxid]) +
      geom_step(aes(x = iteration, y = mean_test, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_test - se_test, ymax = mean_test + se_test, fill = optimizer, linetype = reshuffle), alpha = 0.1, colour = NA) +
      facet_wrap(~ data_id + train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line=FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Test Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("subpath", subpath, gsub("valid_type", valid_type, "plots/subpath/bo_valid_type_test.pdf")), width = 9, height = 15)
  }



  ### improvement and relative performance
  agg = dat[, .(
    mean_log_rel_test = mean(log(rel_test)),
    se_log_rel_test = sd(log(rel_test)) / sqrt(.N),
    mean_improvement_rel_test_baseline = mean(improvement_rel_test_baseline),
    se_improvement_rel_test_baseline = sd(improvement_rel_test_baseline) / sqrt(.N)
  ), by = .(iteration, train_valid_size, resampling, reshuffle, method, optimizer, metric)]

  ggplot(data = agg) +
    geom_step(aes(x = iteration, y = mean_improvement_rel_test_baseline, colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = optimizer, linetype = reshuffle), colour = NA, alpha = 0.1) +
    facet_wrap(~ train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Test Improvement") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_imp.pdf"), width = 9, height = 3)

  ggplot(data = agg) +
    geom_step(aes(x = iteration, y = exp(mean_log_rel_test), colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = optimizer, linetype = reshuffle), colour = NA, alpha = 0.1) +
    geom_hline(yintercept = 1, linetype = 3) +
    facet_wrap(~ train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Relative Test Performance") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_rel.pdf"), width = 9, height = 3)



  ### improvement and relative performance classifier level
  agg = dat[, .(
    mean_log_rel_test = mean(log(rel_test)),
    se_log_rel_test = sd(log(rel_test)) / sqrt(.N),
    mean_improvement_rel_test_baseline = mean(improvement_rel_test_baseline),
    se_improvement_rel_test_baseline = sd(improvement_rel_test_baseline) / sqrt(.N)
  ), by = .(iteration, classifier, train_valid_size, resampling, reshuffle, method, metric, optimizer)]

  classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
  for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
    classifierxid = classifierxids[[classifierx]]
    ggplot(data = agg[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = mean_improvement_rel_test_baseline, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymax = mean_improvement_rel_test_baseline + se_improvement_rel_test_baseline, ymin = mean_improvement_rel_test_baseline - se_improvement_rel_test_baseline, fill = optimizer, linetype = reshuffle), colour = NA, alpha = 0.1) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Test Improvement") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_imp.pdf")), width = 9, height = 3)

    ggplot(data = agg[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = exp(mean_log_rel_test), colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymax = exp(mean_log_rel_test + se_log_rel_test), ymin = exp(mean_log_rel_test - se_log_rel_test), fill = optimizer, linetype = reshuffle), colour = NA, alpha = 0.1) +
      geom_hline(yintercept = 1, linetype = 3) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Relative Test Performance") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_rel.pdf")), width = 9, height = 3)
  }



  ### ranks
  dat[, rank := frank(test, ties.method = "average"), by = .(iteration, data_id, seed, classifier, metric, train_valid_size)]
  ranks = dat[, .(mean_rank = mean(rank), se_rank = sd(rank) / sqrt(.N)), by = .(iteration, metric, resampling, train_valid_size, reshuffle, method, optimizer)]
  ggplot(data = ranks) +
    geom_step(aes(x = iteration, y = mean_rank, colour = optimizer, linetype = reshuffle)) +
    geom_stepribbon(aes(x = iteration, ymin = mean_rank - se_rank, ymax = mean_rank + se_rank, fill = optimizer, linetype = reshuffle), colour = NA, alpha = 0.1) +
    facet_wrap(~ train_valid_size, scales = "free", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
    scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
    scale_color +
    scale_fill +
    xlab("No. HPC Evaluations") +
    ylab("Mean Rank (Test Performance)") +
    labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave(gsub("valid_type", valid_type, "plots/bo_valid_type_rel_ranks.pdf"), width = 9, height = 3)



  ### ranks classifier level
  ranks = dat[, .(mean_rank = mean(rank), se_rank = sd(rank) / sqrt(.N)), by = .(iteration, classifier, metric, resampling, train_valid_size, reshuffle, method, optimizer)]
  classifierxids = list("catboost" = "CatBoost", "xgboost" = "XGBoost", "funnel_mlp" = "Funnel MLP", "logreg" = "Elastic Net")
  for (classifierx in c("catboost", "xgboost", "funnel_mlp", "logreg")) {
    classifierxid = classifierxids[[classifierx]]
    ggplot(data = ranks[classifier == classifierx]) +
      geom_step(aes(x = iteration, y = mean_rank, colour = optimizer, linetype = reshuffle)) +
      geom_stepribbon(aes(x = iteration, ymin = mean_rank - se_rank, ymax = mean_rank + se_rank, fill = optimizer, linetype = reshuffle), colour = NA, alpha = 0.1) +
      facet_wrap(~ train_valid_size, scales = "free_y", ncol = 3, labeller = label_wrap_gen(multi_line = FALSE)) +
      scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 250)) +
      scale_color +
      scale_fill +
      xlab("No. HPC Evaluations") +
      ylab("Mean Rank (Test Performance)") +
      labs(color = "Optimizer", fill = "Optimizer", linetype = "Reshuffling") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(gsub("classifierx", classifierx, gsub("valid_type", valid_type, "plots/bo_classifierx_valid_type_rel_ranks.pdf")), width = 9, height = 3)
  }
}
