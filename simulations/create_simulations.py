if __name__ == "__main__":
    import os

    file = "run_simulations.sh"

    with open(file, "+w") as f:
        for alpha in [0.5, 1, 5, 10]:
            for lengthscale in [0.1, 0.5, 1, 5]:
                f.write(
                    f"python3 simulate.py --alpha {alpha} --lengthscale {lengthscale}\n"
                )

    os.chmod(file, 0o755)
