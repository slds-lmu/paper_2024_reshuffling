library(tidyverse)
library(latex2exp)

## simulating from Gaussian process
sim_gp <- function(n, mu, K, seed = 2) {
  set.seed(seed)
  sq <- 1:n / (n + 1)
  Sig <- outer(sq, sq, K)
  # suboptimal numerical hack
  eig <- eigen(Sig)
  eig$values[eig$values <= 0] <- 0
  Sig_sqrt <- eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors)
  L <- Sig_sqrt %*% rnorm(n) + mu(sq)
}


n <- 50  # number of hyperparameters
mu <- function(lambda) (lambda - 0.5)^2 / 0.1  # true performance function
# covariance kernels
K <- function(x, y, l = 0.01) exp(- (x - y)^2 / l) # synchronous
K_shuffled <- function(x, y, tausq = 0.1) tausq * K(x, y) + (1 - tausq) * (x == y)


# how the validation losses look like
par(mfrow = c(1, 2))
plot(1:n / (n + 1), sim_gp(n, mu, K), type = "l",
     xlab = "lambda", ylab = "L(lambda)", main = "synchronous")
plot(1:n / (n + 1), sim_gp(n, mu, K_shuffled), type = "l",
     xlab = "lambda", ylab = "L(lambda)", main = "shuffled")


## illustration of the effect of tau
dat <- tibble(lambda = 1:n / (n + 1), 
              mu = mu(lambda), 
              sim_sync =  sim_gp(n, mu, K),
              sim_shuffled = sim_gp(n, mu, K_shuffled))

dat |> 
  pivot_longer(-lambda, values_to = "val", names_to = "type") |>
  mutate(type = factor(type, levels = c("mu", "sim_sync", "sim_shuffled"))) |>
  ggplot(aes(lambda, val, color = type, linetype = type)) +
  geom_line() +
  theme_minimal() +
  scale_linetype_manual(
    values = c(1, 1, 2),
    labels = unname(TeX(c("True", "Empirical ($\\tau = 1$$)", "Empirical ($\\tau = 0.3$$)")))
  ) +
  scale_color_manual(
    values = c("black", 4, 4),
    labels = unname(TeX(c("True", "Empirical ($\\tau = 1$$)", "Empirical ($\\tau = 0.3$$)")))
  ) +
  theme(legend.position = "bottom") +
  labs(x = unname(TeX("$\\lambda$")), y = TeX("Loss Surface"), linetype = "Type", color = "Type")
ggsave("plots/lambda-opt-vs-tau-illustration-1.pdf", width = 4, height = 3.7)



K <- function(x, y, l = 4) exp(- (x - y)^2 / l) # synchronous
K_shuffled <- function(x, y, tausq = 0.1) tausq * K(x, y) + (1 - tausq) * (x == y)

dat <- tibble(lambda = 1:n / (n + 1), 
              mu = mu(lambda), 
              sim_sync =  sim_gp(n, mu, K, 5),
              sim_shuffled = sim_gp(n, mu, K_shuffled, 5))

dat |> 
  pivot_longer(-lambda, values_to = "val", names_to = "type") |>
  mutate(type = factor(type, levels = c("mu", "sim_sync", "sim_shuffled"))) |>
  ggplot(aes(lambda, val, color = type, linetype = type)) +
  geom_line() +
  theme_minimal() +
  scale_linetype_manual(
    values = c(1, 1, 2),
    labels = unname(TeX(c("True", "Empirical ($\\tau = 1$$)", "Empirical ($\\tau = 0.3$$)")))
  ) +
  scale_color_manual(
    values = c("black", 4, 4),
    labels = unname(TeX(c("True", "Empirical ($\\tau = 1$$)", "Empirical ($\\tau = 0.3$$)")))
  ) +
  theme(legend.position = "bottom") +
  labs(x = unname(TeX("$\\lambda$")), y = TeX("Loss Surface"), linetype = "Type", color = "Type")

ggsave("plots/lambda-opt-vs-tau-illustration-2.pdf", width = 4, height = 3.7)

