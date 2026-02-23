# Algorithm 1: Local Updating Procedure

## Description

This algorithm performs a localized Hamilton--Jacobi (HJ) update to
correct detected leaking corners while maintaining computational
efficiency.

------------------------------------------------------------------------

## Inputs

-   `V̂(·, ·)` --- Approximated value function\
-   `L̂(·)` --- Detected leaking corner set\
-   `Z` --- State grid\
-   `tlist = [t, t + δ, ..., 0]` --- Time steps

## Output

-   `V̌(·, ·)` --- Corrected value function

------------------------------------------------------------------------

## Pseudocode

    s ← 0                     ▷ Backward computation
    V̌(·, 0) ← V̂(·, 0)
    Frontier ← ∅
    nextFrontier ← ∅
    visited ← ∅

    while s > t do
        for z ∈ Z do
            if z ∈ L̂(s) then
                updateValue(z, s, δ, Frontier)
            else
                V̌(z, s − δ) ← V̂(z, s − δ)
            end if
        end for

        visited ← L̂(s)
        Frontier ← Frontier \ visited

        while Frontier ≠ ∅ do
            for z ∈ Frontier do
                updateValue(z, s, δ, nextFrontier)
            end for

            visited ← visited ∪ Frontier
            Frontier ← nextFrontier \ visited
            nextFrontier ← ∅
        end while

        s ← s − δ
    end while

------------------------------------------------------------------------

## Subroutine

    procedure updateValue(z, s, δ, Frontier):
        V̌(z, s − δ) ← HJ_Update(V̌(z, s))     ▷ Equation (3)

        if V̌(z, s − δ) ≠ V̂(z, s − δ) then
            Frontier ← Frontier ∪ neighbor(z)
        end if
    end procedure

------------------------------------------------------------------------

## Key Idea

-   Detect leaking corners.
-   Apply localized HJ updates only to affected regions.
-   Propagate corrections outward until consistency is restored.
-   Maintain computational efficiency by avoiding full-grid
    recomputation.
