import sys, numpy as np; sys.path.insert(0, "codebase")
from fisher.se3 import (compute_localization_orientation_crlb, predict_se3_rank_from_symmetry,
                        compare_observed_and_predicted_se3_rank)
from fisher.fusion import compute_modality_fusion_crlb_from_fisher_matrices, sigma_xy_from_fisher
from fisher.time_allocation import _scalarize_crlb, compute_loewner_dominance

print("=== (A1) SE(3) rank-from-symmetry PREDICTION algebra (the theorem) ===")
for symdim, er, es in [(3,0,3),(1,2,5),(0,3,6)]:
    p=predict_se3_rank_from_symmetry(symdim)
    ok=p["predicted_rotational_rank"]==er and p["predicted_se3_rank"]==es
    print(f"  sym_dim={symdim}: rot_rank={p['predicted_rotational_rank']}(exp {er}) se3_rank={p['predicted_se3_rank']}(exp {es}) {'OK' if ok else 'FAIL'}")

print("\n=== (A2) SE(3) theorem in action: rotational symmetry -> rotation axes singular ===")
n=41;c=n//2;yy,xx=np.indices((n,n),float)
blob=np.exp(-((xx-c)**2+(yy-c)**2)/(2*4.0**2))
ren={"centre":blob,"z_plus":blob*1.03,"z_minus":blob*0.97}
for ax in ["rx","ry","rz"]: ren[ax+"_plus"]=blob; ren[ax+"_minus"]=blob
cr=compute_localization_orientation_crlb(ren,1.0,100.0,50.0,0.05)
cmp=compare_observed_and_predicted_se3_rank(cr,{"continuous_rotational_symmetry_dim":3})
print(f"  sphere: rank={cr['rank']} axes_singular={cr['axes_singular']} matches_prediction={cmp['rank_matches_symmetry_prediction']} nullity_bound_ok={cmp['satisfies_symmetry_nullity_bound']}")

print("\n=== (B) Fisher fusion additivity + complementary recovery (corollary) ===")
F1=np.array([[4.,0.],[0.,1e-12]]); F2=np.array([[1e-12,0.],[0.,9.]])
r=compute_modality_fusion_crlb_from_fisher_matrices({"A":F1,"B":F2})
print(f"  single: A sigma_xy={sigma_xy_from_fisher(F1):.2e}(y sing) B={sigma_xy_from_fisher(F2):.2e}(x sing)")
print(f"  fused sigma_x={r['fusion_sigma_x_nm']:.4f}(=0.5) sigma_y={r['fusion_sigma_y_nm']:.4f}(=0.333) sigma_xy={r['fusion_sigma_xy_nm']:.4f}")
print(f"  == inv(F1+F2) sigma_xy={sigma_xy_from_fisher(F1+F2):.4f}  finite-from-two-singular={np.isfinite(r['fusion_sigma_xy_nm'])}")

print("\n=== (C) A/D/E scalarizations + Loewner dominance ===")
F=np.array([[5.,1.],[1.,3.]]); Fi=np.linalg.inv(F)
print(f"  A=tr(Fi):{_scalarize_crlb(F,'A'):.6f} vs {np.trace(Fi):.6f}  D=-logdet:{_scalarize_crlb(F,'D'):.6f} vs {-np.log(np.linalg.det(F)):.6f}  E=1/lmin:{_scalarize_crlb(F,'E'):.6f} vs {1/np.linalg.eigvalsh(F)[0]:.6f}")
dom=compute_loewner_dominance({"FA":F+2*np.eye(2),"FB":F})
print(f"  Loewner FA=FB+2I: dominates={dom['dominates']}")
