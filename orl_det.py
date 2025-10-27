"""
pca_psa_gpu.py
PCA -> PSA (on PCA-reduced space) -> SVM comparison
GPU acceleration via CuPy if available; otherwise fallback to NumPy (CPU).
Saves results and plots to ./results/.
"""

import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Try to import cupy
USE_CUPY = False
try:
    import cupy as cp
    USE_CUPY = True
except Exception as e:
    cp = None
    USE_CUPY = False

print("USE_CUPY =", USE_CUPY)
if USE_CUPY:
    try:
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        print("GPU:", props['name'].decode('utf-8'), "compute:", props['major'], props['minor'])
    except Exception as e:
        print("Warning: cannot query GPU properties:", e)

# -------------------------
# Settings (user can change)
data_dir = 'att_faces'   # ORL/ATT dataset folder: att_faces/s1/1.pgm ...
num_persons = 40
num_imgs_per_person = 10
imgH, imgW = 112, 92

# PCA reduction before PSA: IMPORTANT — choose reasonably small Lr (e.g., 60~150)
Lr = 100   # reduced dimensionality (tune this). Lr^3 must fit memory.
Kp = 50    # PCA features for classification
Ks = 20    # PSA components to extract (<= Lr)
max_iter_psa = 80
tol_psa = 1e-4

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# -------------------------
# Helper small wrappers for numpy/cupy interchange
def xp_is_cupy():
    return USE_CUPY

def asarray(x):
    return (cp.array(x) if USE_CUPY else np.array(x))

def to_cpu(x):
    if USE_CUPY:
        return cp.asnumpy(x)
    return x

def dot(a,b):
    if USE_CUPY:
        return cp.dot(a,b)
    return np.dot(a,b)

def svd(a, full_matrices=False):
    if USE_CUPY:
        return cp.linalg.svd(a, full_matrices=full_matrices)
    else:
        return np.linalg.svd(a, full_matrices=full_matrices)

def eigh(a):
    if USE_CUPY:
        return cp.linalg.eigh(a)
    else:
        return np.linalg.eigh(a)

def kron(a,b):
    if USE_CUPY:
        return cp.kron(a,b)
    else:
        return np.kron(a,b)

def mean(a, axis=None, keepdims=False):
    if USE_CUPY:
        return cp.mean(a, axis=axis, keepdims=keepdims)
    else:
        return np.mean(a, axis=axis, keepdims=keepdims)

def norm(a):
    if USE_CUPY:
        return cp.linalg.norm(a)
    else:
        return np.linalg.norm(a)

# -------------------------
# 1. Load ORL dataset
print("Loading dataset from", data_dir)
X_list = []
y_list = []
for i in range(1, num_persons + 1):
    for j in range(1, num_imgs_per_person + 1):
        p = os.path.join(data_dir, f"s{i}", f"{j}.pgm")
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        X_list.append(img.flatten().astype(np.float32))
        y_list.append(i)
X = np.stack(X_list, axis=1)  # shape (L_raw, M)
y = np.array(y_list)
L_raw, M = X.shape
print("Loaded X shape:", X.shape)

# 2. Split train/test (5/5 per person)
train_idx = np.zeros(M, dtype=bool)
for i in range(num_persons):
    train_idx[i*10:(i*10+5)] = True
test_idx = ~train_idx
Xtrain = X[:, train_idx].astype(np.float32)
Xtest  = X[:, test_idx].astype(np.float32)
ytrain = y[train_idx]
ytest  = y[test_idx]
print("Train shape:", Xtrain.shape, "Test shape:", Xtest.shape)

# 3. PCA on raw pixels to reduce to Lr — do on CPU for stability (use numpy)
#    We will still optionally perform heavy ops later on GPU (CuPy).
print("\nStage A: PCA reduction to Lr =", Lr)
t0 = time.time()
# center
mu_pixels = np.mean(Xtrain, axis=1, keepdims=True)
Xc = Xtrain - mu_pixels  # L_raw x M_train

# Small trick: compute PCA via SVD on covariance in sample-space (MxM) for speed
C_small = Xc.T @ Xc    # M_train x M_train
U_small, S_small, _ = np.linalg.svd(C_small, full_matrices=False)
# principal components in original space:
pcs = Xc @ U_small[:, :Lr]   # L_raw x Lr
# normalize
pcs = pcs / np.linalg.norm(pcs, axis=0, keepdims=True)

# Project train/test to reduced PCA space (dimension Lr)
Ztrain = pcs.T @ Xc                    # Lr x M_train
Ztest  = pcs.T @ (Xtest - mu_pixels)   # Lr x M_test
t1 = time.time()
print("PCA reduction time: %.3fs" % (t1 - t0))

# 4. Whitening in reduced space (we will do PSA on whitened reduced data)
print("\nStage B: Whitening reduced data (size Lr x M_train)")
t0 = time.time()
# covariance in reduced space
Cov_reduced = (Ztrain @ Ztrain.T) / Ztrain.shape[1]
eigvals, eigvecs = np.linalg.eigh(Cov_reduced)
# numerical safety: clamp small eigenvalues
eigvals[eigvals <= 1e-12] = 1e-12
W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # whitening matrix (Lr x Lr)
B_hat_cpu = W @ Ztrain   # Lr x M_train
t1 = time.time()
print("Whitening time: %.3fs" % (t1 - t0))

# 5. PCA classification baseline (use Kp components from earlier 'pcs' or from reduced)
print("\nStage C: PCA baseline classification (Kp=%d) using SVM (CPU sklearn)" % Kp)
t0 = time.time()
# further reduce to Kp for classifier
Kp_used = min(Kp, Lr)
# Option A: compute eigenfaces in reduced pipeline -> use top Kp basis from pcs*eigvecs etc.
# Simpler: perform SVD on Ztrain to get principal directions (in reduced space)
U_z, S_z, Vt_z = np.linalg.svd(Ztrain, full_matrices=False)
# basis in original (reduced) space: U_z columns
proj_basis = U_z[:, :Kp_used]    # Lr x Kp_used
# project data:
Ytrain_pca = proj_basis.T @ Ztrain
Ytest_pca  = proj_basis.T @ Ztest
# train SVM on CPU
clf_pca = SVC(kernel='linear')
clf_pca.fit(Ytrain_pca.T, ytrain)
ypred_pca = clf_pca.predict(Ytest_pca.T)
acc_pca = accuracy_score(ytest, ypred_pca) * 100
t1 = time.time()
print("PCA classification accuracy: %.2f%%, time %.3fs" % (acc_pca, t1 - t0))

# 6. PSA on reduced whitened data (B_hat_cpu) — heavy ops: try to use CuPy if available
print("\nStage D: PSA on PCA-reduced whitened data (Lr=%d, M=%d)" % (Lr, B_hat_cpu.shape[1]))
# Move data to GPU if available
if USE_CUPY:
    xp = cp
    B_hat = cp.array(B_hat_cpu)   # Lr x M
else:
    xp = np
    B_hat = B_hat_cpu

L = Lr
M_train = B_hat.shape[1]
if Ks > L:
    print("Warning: Ks > Lr. Setting Ks = Lr.")
    Ks = L

# compute coskewness tensor S (L x L x L)
# S = (1/M) * tensor( B_hat * (Khatri-Rao(B_hat, B_hat))^T )
# implement efficient approach: S(i,:,:)= (1/M) * B_hat(i,:) * (B_hat ⊙ B_hat).T  (per paper)
# We'll compute S_tensor as L x L x L (careful with memory)
print("Computing coskewness tensor (this may take some time)...")
t0 = time.time()

# Compute Khatri-Rao product per column: each column k is kron(b_k, b_k) -- but we'll form matrix L^2 x M
# We'll do: B_hat * (B_hat ⊙ B_hat)^T in a memory-friendly vectorized way
# Step: compute elementwise outer contributions via broadcasting and reshape
# Implementation: compute S_tensor by summing over samples: S += r_i ⊗ r_i ⊗ r_i / M

# We'll accumulate S in float32 (or float64 if small)
if USE_CUPY:
    S_tensor = cp.zeros((L, L, L), dtype=cp.float32)
    for m in tqdm(range(M_train), desc="accumulating S (GPU)"):
        r = B_hat[:, m]          # L
        # outer product r ⊗ r ⊗ r
        # First compute r ⊗ r -> L x L, then multiply each element by r (broadcast along dim0)
        # r3 = r[:,None,None] * r[None,:,None] * r[None,None,:]  -> LxLxL
        # But this builds full cube for each m which is heavy; instead add slices:
        S_tensor += r[:, None, None] * r[None, :, None] * r[None, None, :]
    S_tensor = S_tensor / M_train
else:
    S_tensor = np.zeros((L, L, L), dtype=np.float32)
    for m in tqdm(range(M_train), desc="accumulating S (CPU)"):
        r = B_hat[:, m]
        S_tensor += r[:, None, None] * r[None, :, None] * r[None, None, :]
    S_tensor = S_tensor / M_train

t1 = time.time()
print("Finished coskewness tensor: time %.3fs, S_tensor shape %s" % (t1 - t0, str(S_tensor.shape)))

# 7. PSA fixed-point iterations to get U_psa (L x Ks)
print("\nStage E: PSA fixed-point iterations (Ks=%d)" % Ks)
t0 = time.time()
# initialize
if USE_CUPY:
    U_psa = cp.zeros((L, Ks), dtype=cp.float32)
    Pk = cp.eye(L, dtype=cp.float32)
else:
    U_psa = np.zeros((L, Ks), dtype=np.float32)
    Pk = np.eye(L, dtype=np.float32)

for p in range(Ks):
    if USE_CUPY:
        u = cp.random.randn(L).astype(cp.float32)
        u = u / cp.linalg.norm(u)
    else:
        u = np.random.randn(L).astype(np.float32)
        u = u / np.linalg.norm(u)

    for it in range(max_iter_psa):
        # projected vector
        Pu = Pk @ u    # L
        if USE_CUPY:
            # compute Su2 = S_tensor ×1 (Pu) ×2 (Pu)  -> vector of length L
            # vectorized sum: sum over axes 1 and 2
            Su2 = cp.sum(S_tensor * Pu[:, None, None] * Pu[None, :, None], axis=(1,2))
            # normalize
            nrm = cp.linalg.norm(Su2)
            if nrm == 0:
                break
            u_new = Su2 / nrm
            if cp.linalg.norm(u_new - u) < tol_psa:
                u = u_new
                break
            u = u_new
        else:
            Pu_r = Pu.ravel()
            Su2 = np.sum(S_tensor * Pu_r[:, None, None] * Pu_r[None, :, None], axis=(1,2))
            nrm = np.linalg.norm(Su2)
            if nrm == 0:
                break
            u_new = Su2 / nrm
            if np.linalg.norm(u_new - u) < tol_psa:
                u = u_new
                break
            u = u_new

    # store u
    if USE_CUPY:
        U_psa[:, p] = u
        # update Pk = I - U(:,1:p+1) * (U' * U)^{-1} * U'
        Usub = U_psa[:, :p+1]
        invGram = cp.linalg.inv(Usub.T @ Usub)
        Pk = cp.eye(L, dtype=cp.float32) - Usub @ (invGram @ Usub.T)
    else:
        U_psa[:, p] = u
        Usub = U_psa[:, :p+1]
        invGram = np.linalg.inv(Usub.T @ Usub)
        Pk = np.eye(L, dtype=np.float32) - Usub @ (invGram @ Usub.T)

t1 = time.time()
print("PSA extraction time: %.3fs" % (t1 - t0))

# 8. Project train/test to PSA features and perform SVM (note sklearn expects numpy)
print("\nStage F: Classification with PSA features (SVM on CPU)")
if USE_CUPY:
    U_psa_cpu = cp.asnumpy(U_psa)
    B_hat_cpu = cp.asnumpy(B_hat)
else:
    U_psa_cpu = U_psa
    B_hat_cpu = B_hat

Ytrain_psa = U_psa_cpu.T @ B_hat_cpu    # Ks x M_train
# for test: need same whitening and PCA transforms: Ztest -> whiten via W, then project via U_psa
Ztest_cpu = Ztest  # already numpy
W_cpu = W
Bhat_test_cpu = W_cpu @ Ztest_cpu
Ytest_psa = U_psa_cpu.T @ Bhat_test_cpu

# train SVM
t0 = time.time()
clf_psa = SVC(kernel='linear')
clf_psa.fit(Ytrain_psa.T, ytrain)
ypred_psa = clf_psa.predict(Ytest_psa.T)
acc_psa = accuracy_score(ytest, ypred_psa) * 100
t1 = time.time()
print("PSA classification accuracy: %.2f%%, time %.3fs" % (acc_psa, t1 - t0))

# 9. Save results and plots
print("\nSaving results and plots to", results_dir)
np.save(os.path.join(results_dir, "acc_pca.npy"), np.array([acc_pca]))
np.save(os.path.join(results_dir, "acc_psa.npy"), np.array([acc_psa]))

# confusion matrices
cm_pca = confusion_matrix(ytest, ypred_pca)
cm_psa = confusion_matrix(ytest, ypred_psa)

fig, axes = plt.subplots(1,2, figsize=(12,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pca)
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title("PCA Confusion Matrix")
disp = ConfusionMatrixDisplay(confusion_matrix=cm_psa)
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title("PSA Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrices.png"))
plt.close()

# visualize first 10 PCA "components" (in pixel space) by mapping back via pcs (L_raw x Lr)
fig = plt.figure(figsize=(12,4))
for i in range(min(10, Kp_used)):
    ax = fig.add_subplot(2,5,i+1)
    patch = (pcs[:, i]).reshape((imgH,imgW))
    ax.imshow(patch, cmap='gray'); ax.axis('off'); ax.set_title(f"PCA#{i+1}")
plt.suptitle("First PCA basis (reduced space -> pixel domain)")
plt.savefig(os.path.join(results_dir, "pca_basis.png"))
plt.close()

# visualize first 10 PSA components mapped back to pixel space
fig = plt.figure(figsize=(12,4))
for i in range(min(10, Ks)):
    ax = fig.add_subplot(2,5,i+1)
    # PSA basis in reduced space U_psa_cpu[:,i]; map back to pixels via pcs (L_raw x Lr) * coeffs
    back = pcs @ (U_psa_cpu[:, i])
    ax.imshow(back.reshape((imgH,imgW)), cmap='gray'); ax.axis('off'); ax.set_title(f"PSA#{i+1}")
plt.suptitle("First PSA components (mapped to pixel domain)")
plt.savefig(os.path.join(results_dir, "psa_components.png"))
plt.close()

# Print summary
print("\nSummary:")
print("PCA acc: %.2f%%" % acc_pca)
print("PSA acc: %.2f%%" % acc_psa)
print("Results saved to:", results_dir)
