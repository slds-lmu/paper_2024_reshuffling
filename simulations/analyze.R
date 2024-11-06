library(data.table)
library(ggplot2)
files = dir("results/")
results = do.call(rbind, lapply(files, function(file) fread(paste0("results/", file))))
agg = results[, .(mean_y_shuffled = mean(y_shuffled), se_y_shuffled = sd(y_shuffled) / sqrt(.N)), by = .(tau, alpha, lengthscale)]
agg[, m := 2 * alpha]
agg[, Kappa := 1 / (lengthscale ^ 2)]
agg = agg[m > 1]

ggplot() +
  geom_point(aes(x = tau, y = mean_y_shuffled), colour = "darkgrey", data = agg[tau != 1]) +
  geom_errorbar(aes(x = tau, ymin = mean_y_shuffled - se_y_shuffled, ymax = mean_y_shuffled + se_y_shuffled),
                width = 0.1, position = position_dodge(0.05), colour = "darkgrey", data = agg[tau != 1]) +
  geom_point(aes(x = tau, y = mean_y_shuffled), colour = "black", data = agg[tau == 1]) +
  geom_errorbar(aes(x = tau, ymin = mean_y_shuffled - se_y_shuffled, ymax = mean_y_shuffled + se_y_shuffled),
                width = 0.1, position = position_dodge(0.05), colour = "black", data = agg[tau == 1]) +
  facet_wrap(~ m + Kappa, scales = "free_y", ncol = 4,
             labeller = labeller(
               m = label_both,
               Kappa = function(value) paste0("κ: ", value)
             )) +
  xlab(expression(tau)) +
  ylab(expression(mu(hat(lambda)))) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/simulation_results.pdf", width = 8, height = 5, device = cairo_pdf)
