#!/usr/bin/env python3
"""Estimate the scattering-Gaussian product used for the signal-side model.

This is the prior-facing entry point for ``estimate_scattering_covariance``.
It intentionally shares the complete implementation so that preprocessing,
STL normalization/reference state, covariance augmentation, masking, and file
formats remain identical.  Use ``--compute-bias`` together with
``--bias-reference-map`` and ``--noise-input`` to additionally save

    bias = mean_noise(phi(reference + noise)) - phi(reference).
"""

from estimate_scattering_covariance import main


if __name__ == "__main__":
    main()
