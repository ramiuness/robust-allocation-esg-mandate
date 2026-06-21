# Robust End-to-End Portfolio Allocation

End-to-end differentiable portfolio optimization extending the framework of Costa & Iyengar (2022). Integrates a linear return prediction layer with a differentiable convex optimization layer, backpropagating task loss jointly through both stages.

**Optimization layers**
- `base_mod` — pure return maximization
- `base_rom` — estimation-robust SOCP; uncertainty ellipsoid on the predicted mean 
- `nominal`, `tv`, `hellinger` — distributionally robust layers with empirical, TV-ball, and Hellinger-ball ambiguity sets

**Training modes**
- `pred_then_opt` (PO) — fixed OLS prediction weights, optimization layer only
- `e2e_net` (SPO) — end-to-end learned prediction weights via gradient backpropagation through the optimization layer

## Structure

```
src/e2edro/     core module (models, loss functions, data loading, portfolio classes)
scripts/        plotting utilities and data extraction helpers
```

## Reference

Costa, G. and Iyengar, G. N. (2022). *Distributionally Robust End-to-End Portfolio Construction.* arXiv:2206.05134 [q-fin.CP].

Original implementation: [Iyengar-Lab/E2E-DRO](https://github.com/Iyengar-Lab/E2E-DRO)
