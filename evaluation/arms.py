"""Named generator configurations Track 2 evaluates. Each arm is (generator name, hyperparameters)."""

ARMS = {
    "independent_marginals": ("independent_marginals", {}),
    "gaussian_copula": ("gaussian_copula", {}),
    "gaussian_copula_shrink025": ("gaussian_copula", {"shrinkage": 0.25}),  # config Track 1 carried into Phase 3 (ADR-024)
    "ctgan": ("ctgan", {}),  # 300 epochs, the least-bad CTGAN config (ADR-024)
    "tvae": ("tvae", {}),  # 300 epochs; memorizes, so it is the membership-attack positive control (ADR-024)
    # Added for the final comparison (ADR-029): the tuned configs from Track 1's Phase 3 sweep, and the
    # diffusion family, which the Phase 3 evaluation predates.
    "gaussian_copula_shrink010": ("gaussian_copula", {"shrinkage": 0.10}),  # tuned statistical baseline
    "ctgan_small": ("ctgan", {"epochs": 300, "embedding_dim": 32,
                              "generator_dim": [64, 64], "discriminator_dim": [64, 64]}),  # tuned GAN arm
    "tabular_diffusion": ("tabular_diffusion", {}),  # 3000 steps: best neural fidelity, memorizes
    "tabular_diffusion_reg": ("tabular_diffusion", {"steps": 1000, "dropout": 0.2}),  # only guard-passing setting
    # Post-course extension arms (improvement plan item 4), NOT part of the nine-arm reported benchmark:
    # differentially private baselines at two budgets, so the benchmark has at least one formal guarantee.
    "dp_marginals_eps1": ("dp_marginals", {"epsilon": 1.0}),
    "dp_marginals_eps8": ("dp_marginals", {"epsilon": 8.0}),
    "dp_copula_eps1": ("dp_copula", {"epsilon": 1.0}),
    "dp_copula_eps8": ("dp_copula", {"epsilon": 8.0}),
}

# The nine arms the course benchmark reports, so "all arms" never silently means the extension set.
REPORTED_ARMS = [
    "independent_marginals", "gaussian_copula", "gaussian_copula_shrink025", "gaussian_copula_shrink010",
    "ctgan", "ctgan_small", "tvae", "tabular_diffusion", "tabular_diffusion_reg",
]


def resolve(arm):
    return ARMS.get(arm, (arm, {}))
