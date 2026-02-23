# Threshold Strategy for Fixing Leaking Corners

*A Conceptual Explanation*

------------------------------------------------------------------------

## 1. The Core Problem

When using dimensionality reduction (such as subsystem decomposition) in
Hamilton--Jacobi (HJ) reachability, we:

1.  Compute low-dimensional value functions independently.

2.  Reconstruct a full-dimensional approximation:

    -   **Intersection case (liveness)**\
        V̂(z, t) = max_i V_i(z, t)

    -   **Union case (safety)**\
        V̂(z, t) = min_i V_i(z, t)

However, subsystem-optimal controls may **not satisfy the original
coupled control constraint**.\
This mismatch creates inaccuracies known as:

> **Leaking corners** --- states where the reconstructed value function
> differs from the true full-dimensional value function.

Formally:

L(t) = { z : V(z,t) ≠ V̂(z,t) }

------------------------------------------------------------------------

## 2. Why Leaking Corners Occur

Each subsystem independently chooses its optimal control.

But the original system enforces a **joint control constraint**.

When both subsystems simultaneously apply their own optimal controls:

-   The combined control may violate the constraint.
-   The reconstructed value becomes overly optimistic or pessimistic.
-   The error appears near regions where subsystem values are close.

These regions form connected sets called **islands** of leaking corners.

------------------------------------------------------------------------

## 3. The Key Insight Behind the Threshold Strategy

The main observation:

> Leaking corners occur when subsystem value functions are close to each
> other.

Define the value difference:

V_d(z, t) = \|V₁(z, t) − V₂(z, t)\|

If this difference is small enough, then both subsystems are "competing"
to dominate the reconstruction --- which indicates potential constraint
conflict.

------------------------------------------------------------------------

## 4. The Threshold Condition

We introduce a threshold Δ such that:

\|V₁(z,t) − V₂(z,t)\| \< Δ

⇒ z is classified as a leaking corner.

### How Δ is Chosen

Δ is derived from allowable control deviations:

Δ = \|Ṽ\_i − V_i\|

where: - V_i is the subsystem value under its optimal control, - Ṽ\_i is
the value under an allowable (constraint-satisfying) control pair.

This guarantees that detection is tied directly to constraint violation.

------------------------------------------------------------------------

## 5. Detection Step

For every state z:

1.  Compute subsystem values V₁ and V₂.
2.  Compute their difference.
3.  If the difference is below Δ → mark z as leaking.

This produces an approximated leaking set L̂(t).

This detection: - Requires no additional full-dimensional solve. - Works
for scalar and vector control inputs. - Is compatible with decomposed
computations.

------------------------------------------------------------------------

## 6. Correction Step (Local Updating)

Once leaking regions are detected:

1.  Initialize the approximated value function V̂.
2.  Apply full HJ updates **only within detected islands**.
3.  Propagate corrections outward until consistency is restored.

This avoids recomputing the entire grid.

------------------------------------------------------------------------

## 7. Why It Works

The strategy works because:

-   Constraint violation is largest where subsystem values coincide.
-   Moving away from those points reduces violation continuously.
-   Therefore, islands are localized.
-   Local re-solving restores exactness.

The final corrected value function:

-   Matches the ground truth.
-   Preserves computational efficiency.
-   Avoids full high-dimensional recomputation.

------------------------------------------------------------------------

## 8. Summary

The threshold strategy:

1.  Detects leaking corners using a value-difference test.
2.  Uses a theoretically justified threshold Δ.
3.  Applies localized HJ updates only where needed.
4.  Restores exact reachability guarantees.
5.  Maintains scalability for high-dimensional systems.

It transforms dimensionality reduction from an approximate method into
an **efficient yet exact computational framework**.

------------------------------------------------------------------------

*End of explanation.*
