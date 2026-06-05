import sys, numpy as np
sys.path.insert(0, "codebase")
from mie_scattering import mie_an_bn, mie_S1_S2_from_coefficients
import miepython

print("miepython", miepython.__version__)
print("miepython attrs:", [a for a in dir(miepython) if not a.startswith("_")][:20])

m = complex(1.59, 0.0)   # polystyrene-ish in vacuum
x = 3.0
mus = np.array([1.0, 0.5, 0.0, -0.5, -1.0])

a_n, b_n = mie_an_bn(m, x)
S1_s, S2_s = mie_S1_S2_from_coefficients(a_n, b_n, mus)

# miepython v3 API discovery
fn = None
for name in ["mie_S1_S2","S1_S2","mie_s1_s2"]:
    if hasattr(miepython, name):
        fn = getattr(miepython, name); print("using miepython.%s"%name); break
if fn is None:
    print("no S1_S2 fn; attrs=", dir(miepython)); sys.exit(0)

try:
    S1_m, S2_m = fn(m, x, mus)
except Exception as e:
    print("call form A failed:", e)
    S1_m, S2_m = fn(m, x, mus, norm='bohren')

S1_m = np.asarray(S1_m); S2_m = np.asarray(S2_m)
print("\n mu      |S2_syn|        |S2_mie|        ratio(syn/mie)")
for i,mu in enumerate(mus):
    r = abs(S2_s[i])/abs(S2_m[i]) if abs(S2_m[i])>0 else float('nan')
    print("%+.2f   %12.6e   %12.6e   %.6f"%(mu, abs(S2_s[i]), abs(S2_m[i]), r))

# try a normalization-aware comparison: bohren norm
if 'norm' in fn.__doc__.lower() if fn.__doc__ else False:
    pass
print("\nratios consistent? (constant ratio => pure normalization difference)")
ratios = np.abs(S2_s)/np.abs(S2_m)
print("ratio mean=%.6f std=%.3e  -> %s"%(ratios.mean(), ratios.std(),
      "CONSTANT (normalization only)" if ratios.std()/ratios.mean()<1e-6 else "NON-constant"))
