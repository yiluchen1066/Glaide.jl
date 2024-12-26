using Unitful

# fundamental dimensionless constants
const n = 3
const E = 0.25

# dimensional physical constants
const ρ  = 910.0u"kg/m^3"
const g  = 9.81u"m/s^2"
const A  = 2.5e-24u"Pa^-3*s^-1"
const As = 1.0e-22u"Pa^-3*s^-1*m"

ustrip(uconvert(u"hm^-3*yr^-1", (ρ * g)^n * E * A))

# derived dimensional constants
ρgn   = uconvert(u"MPa^3/hm^3", (ρ * g)^n)
ρgnA  = uconvert(u"hm^-3*yr^-1", ρgn * E * A)
ρgnAs = uconvert(u"hm^-2*yr^-1", ρgn * As)

Deff(ρgnA, ρgnAs, ∇S, H, n) = (ρgnA * H^(n + 2) * 2 / (n + 2) + ρgnAs * H^(n + 1)) * ∇S^(n - 1)

Deff(ρgnA, ρgnAs, 0.1, 1.0u"hm", n)
Deff(ρgnA, ρgnAs, 0.01, 1.0u"hm", n)
Deff(ρgnA, ρgnAs, 0.001, 9.0u"hm", n)

uconvert(u"MPa^-3*yr^-1", A)
uconvert(u"MPa^-3*yr^-1*hm", As)

function q(ρgnA, ρgnAs, ∇S, H1, H2, n, dx)
    return (ρgnA * 2 / (n + 2) / (n + 3) * (H2^(n + 3) - H1^(n + 3)) / dx +
            ρgnAs / (n + 2) * (H2^(n + 2) - H1^(n + 2)) / dx) * ∇S^(n - 1)
end

q(ρgnA, ρgnAs, 0.1, 1.0u"hm", 1.01u"hm", n, 0.25u"hm")
q(ρgnA, ρgnAs, 0.01, 1.0u"hm", 1.01u"hm", n, 0.25u"hm")
q(ρgnA, ρgnAs, 0.001, 10.0u"hm", 10.01u"hm", n, 0.25u"hm")
