"""Level-B validation of lateral Fisher/CRLB vs closed-form Gaussian-signal bound.
Closed form: F_xx = A^2 * pi / (2 sigma^2 p^2)  ->  sigma_x = (p*sigma/A)*sqrt(2/pi).
No production-code changes."""
import sys, numpy as np; sys.path.insert(0,"codebase")
from fisher.lateral import compute_localization_crlb, compute_fisher_information

def gaussian(A, s_px, n):
    c=(n-1)/2.0
    y,x=np.indices((n,n),dtype=float)
    r2=(x-c)**2+(y-c)**2
    return A*np.exp(-r2/(2.0*s_px**2))

def predicted_sigma_nm(A, var, p_nm):
    return p_nm*np.sqrt(var)/A*np.sqrt(2.0/np.pi)

print("=== Absolute closed-form check (A=1, var=1, p=100nm, n=121) ===")
for s_px in [3.0, 4.0, 6.0, 8.0]:
    A, var, p = 1.0, 1.0, 100.0
    C=gaussian(A, s_px, 121)
    res=compute_localization_crlb(C, var, p)
    pred=predicted_sigma_nm(A,var,p)
    sx,sy=float(res["sigma_x_nm"]),float(res["sigma_y_nm"])
    F=np.asarray(res["fisher_matrix"])
    print(f"  s={s_px}px: sigma_x={sx:.3f} sigma_y={sy:.3f} nm | pred={pred:.3f} | "
          f"rel_err={abs(sx/pred-1)*100:.2f}% | F_xy/F_xx={F[0,1]/F[0,0]:.1e} rank={res.get('rank')}")

print("\n=== Scaling: sigma_x ∝ sigma_noise / A (s=5px, p=100nm) ===")
s_px=5.0; p=100.0
for (A,var) in [(1.0,1.0),(2.0,1.0),(1.0,4.0),(0.5,1.0),(1.0,0.25)]:
    C=gaussian(A,s_px,121)
    sx=float(compute_localization_crlb(C,var,p)["sigma_x_nm"])
    pred=predicted_sigma_nm(A,var,p)
    print(f"  A={A} var={var}: sigma_x={sx:.3f} pred={pred:.3f} rel_err={abs(sx/pred-1)*100:.2f}%")

print("\n=== width-independence (pred has no s): A=1,var=1,p=100 ===")
preds=set()
for s_px in [2,3,5,9,14]:
    C=gaussian(1.0,float(s_px),161)
    sx=float(compute_localization_crlb(C,1.0,100.0)["sigma_x_nm"])
    print(f"  s={s_px:>2}px -> sigma_x={sx:.3f} nm")
