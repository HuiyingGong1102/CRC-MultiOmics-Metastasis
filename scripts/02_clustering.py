"""Step 02: reproducible four-group linear AR(1) protein clustering.
Adapted from AR1_linear.py. Preserve its AdamW, 25 updates per EM step,
learning rate 1e-5, tolerance 1e-4, and 10000-step limit.
BIC uses the final observed mixture likelihood; legacy Q BIC is also saved.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_SEED = 0

def mahalanobis(x, center, cov):
    x_cen = x - center
    sol = torch.linalg.solve(cov, x_cen.T).T
    dist = torch.sum(x_cen * sol, dim=1)
    return dist

def dmvnorm(x, mean, sigma, log=True):
    distval = mahalanobis(x, mean, sigma)
    L = torch.linalg.cholesky(sigma)
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))

    d = x.shape[1]
    log2pi = torch.log(torch.tensor(2 * np.pi, device=x.device, dtype=torch.float32))

    logretval = -0.5 * (d * log2pi + logdet + distval)

    if log:
        return logretval
    else:
        return torch.exp(logretval)

def linear_equation(x, linear_par):
    result = linear_par[:, 0][:, None] + linear_par[:, 1][:, None] * x
    return result


def linear_equation_base(x, y):
    x = np.array(x, dtype=float).reshape(-1, 1)
    y = np.array(y, dtype=float)

    model = LinearRegression().fit(x, y)
    intercept = model.intercept_
    slope = model.coef_[0]

    return intercept, slope


def logsumexp(v):
    vm = torch.max(v)
    return torch.log(torch.sum(torch.exp(v - vm))) + vm


def get_SAD1_covmatrix(par, n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    par = torch.as_tensor(par, dtype=torch.float32, device=device)

    phi = par[0]
    gamma = par[1]

    idx = torch.arange(n, dtype=torch.float32, device=device)
    diff_matrix = torch.abs(idx[:, None] - idx[None, :])

    den = (1.0 - phi ** 2).clamp(min=1e-12)
    sigma = (phi ** diff_matrix) * (gamma ** 2 / den)

    sigma = (sigma + sigma.T) / 2

    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-6
    sigma = sigma + eps * torch.eye(n, dtype=torch.float32, device=device)

    return sigma


def get_sixSAD1(par, n1, n2, n3, n4):
    sig1 = get_SAD1_covmatrix(par[0:2], n1)
    sig2 = get_SAD1_covmatrix(par[2:4], n2)
    sig3 = get_SAD1_covmatrix(par[4:6], n3)
    sig4 = get_SAD1_covmatrix(par[6:8], n4)

    n = n1 + n2 + n3 + n4
    sig = torch.zeros((n, n), dtype=torch.float32).to(device)

    sig[0:n1, 0:n1] = sig1
    sig[n1:n1 + n2, n1:n1 + n2] = sig2
    sig[n1 + n2:n1 + n2 + n3, n1 + n2:n1 + n2 + n3] = sig3
    sig[n1 + n2 + n3:n1 + n2 + n3 + n4, n1 + n2 + n3:n1 + n2 + n3 + n4] = sig4

    return sig

def get_six_par_int(X, k, times1, times2, times3, times4, n1, n2, n3, n4):
    n, d = X.shape

    cov_int = [0.595141, 1.3078251,
               0.44131, 1.2618071,
               0.5778005, 1.4084324,
               0.4108889, 1.0231676]#par1

    init_cluster = KMeans(n_clusters=k,
                          init='k-means++',
                          n_init=10, random_state=RUN_SEED
                          ).fit(X.cpu())
    labels = init_cluster.labels_

    prob = np.bincount(labels) / n
    times1 = times1.cpu().numpy()
    times2 = times2.cpu().numpy()
    times3 = times3.cpu().numpy()
    times4 = times4.cpu().numpy()

    fit1 = np.array([linear_equation_base(times1, init_cluster.cluster_centers_[c, :n1]) for c in range(k)]).T
    fit2 = np.array([linear_equation_base(times2, init_cluster.cluster_centers_[c, n1:n1 + n2]) for c in range(k)]).T
    fit3 = np.array([linear_equation_base(times3, init_cluster.cluster_centers_[c, n1 + n2:n1 + n2 + n3]) for c in range(k)]).T
    fit4 = np.array([linear_equation_base(times4, init_cluster.cluster_centers_[c, n1 + n2 + n3:n1 + n2 + n3 + n4]) for c in range(k)]).T
    return_obj = {
        'initial_cov_params': cov_int,
        'initial_mu_params': np.hstack((fit1, fit2, fit3, fit4)).flatten(),
        'initial_probibality': prob
    }

    return return_obj


def sixQ_function(par, prob_log, omega_log, X, k, n1, n2, n3, n4, times1, times2, times3, times4):
    n = X.shape[0]
    n1, n2, n3, n4, k = map(int, [n1, n2, n3, n4, k])

    X1 = X[:, :n1]
    X2 = X[:, n1:n1 + n2]
    X3 = X[:, n1 + n2:n1 + n2 + n3]
    X4 = X[:, n1 + n2 + n3:n1 + n2 + n3 + n4]

    par_mu = par[8:]

    cov1 = get_SAD1_covmatrix(par[:2], n1)
    cov2 = get_SAD1_covmatrix(par[2:4], n2)
    cov3 = get_SAD1_covmatrix(par[4:6], n3)
    cov4 = get_SAD1_covmatrix(par[6:8], n4)

    mu1_mat = torch.column_stack((par_mu[:k], par_mu[4 * k:5 * k]))
    mu2_mat = torch.column_stack((par_mu[k:2 * k], par_mu[5 * k:6 * k]))
    mu3_mat = torch.column_stack((par_mu[2 * k:3 * k], par_mu[6 * k:7 * k]))
    mu4_mat = torch.column_stack((par_mu[3 * k:4 * k], par_mu[7 * k:8 * k]))

    mu1 = linear_equation(times1, mu1_mat)
    mu2 = linear_equation(times2, mu2_mat)
    mu3 = linear_equation(times3, mu3_mat)
    mu4 = linear_equation(times4, mu4_mat)

    mvn_log1 = torch.stack([dmvnorm(X1, mu1[i], cov1, True) for i in range(k)], dim=1)
    mvn_log2 = torch.stack([dmvnorm(X2, mu2[i], cov2, True) for i in range(k)], dim=1)
    mvn_log3 = torch.stack([dmvnorm(X3, mu3[i], cov3, True) for i in range(k)], dim=1)
    mvn_log4 = torch.stack([dmvnorm(X4, mu4[i], cov4, True) for i in range(k)], dim=1)

    mvn_log = mvn_log1 + mvn_log2 + mvn_log3 + mvn_log4
    tmp = prob_log + mvn_log - omega_log
    Q = -torch.sum(tmp * torch.exp(omega_log))

    return Q


def sixfun_clu(data, k, Time, n1, n2, n3, n4, initial_pars=None,iter_max=10000):

    d, n = data.shape
    epsilon = 10000
    iter = 0

    X1 = data[:, :n1]
    X2 = data[:, n1:n1 + n2]
    X3 = data[:, n1 + n2:n1 + n2 + n3]
    X4 = data[:, n1 + n2 + n3:n1 + n2 + n3 + n4]

    times1 = Time[:,:n1]
    times2 = Time[:,n1:n1 + n2]
    times3 = Time[:,n1 + n2:n1 + n2 + n3]
    times4 = Time[:,n1 + n2 + n3:n1 + n2 + n3 + n4]

    if initial_pars is None:
        par_int = get_six_par_int(data, k, times1, times2, times3, times4, n1, n2, n3, n4)
        prob_int = np.ones(k) / k
        cov_int = par_int['initial_cov_params']
        mu_int = par_int['initial_mu_params']
        initial_pars = np.hstack((cov_int, mu_int))
    else:
        prob_int = torch.tensor(initial_pars[:k]).to(device)
        initial_pars = initial_pars[k:]

    par = torch.tensor(initial_pars, dtype=torch.float32).to(device)
    par.requires_grad = True
    prob_logi = torch.log(torch.tensor(prob_int, dtype=torch.float32).to(device))

    initial_lr = 1e-5
    optimizer = optim.AdamW([par], lr=initial_lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    while abs(epsilon) > 0.0001 and iter < iter_max:

        par_mui = par[8:]
        cov1i = get_SAD1_covmatrix(par[:2], n1).to(device).detach()
        cov2i = get_SAD1_covmatrix(par[2:4], n2).to(device).detach()
        cov3i = get_SAD1_covmatrix(par[4:6], n3).to(device).detach()
        cov4i = get_SAD1_covmatrix(par[6:8], n4).to(device).detach()

        mu1_mati = torch.column_stack((par_mui[:k], par_mui[4 * k:5 * k])).to(device).detach()
        mu2_mati = torch.column_stack((par_mui[k:2 * k], par_mui[5 * k:6 * k])).to(device).detach()
        mu3_mati = torch.column_stack((par_mui[2 * k:3 * k], par_mui[6 * k:7 * k])).to(device).detach()
        mu4_mati = torch.column_stack((par_mui[3 * k:4 * k], par_mui[7 * k:8 * k])).to(device).detach()

        mu1i = linear_equation(times1, mu1_mati).to(device).detach()
        mu2i = linear_equation(times2, mu2_mati).to(device).detach()
        mu3i = linear_equation(times3, mu3_mati).to(device).detach()
        mu4i = linear_equation(times4, mu4_mati).to(device).detach()

        mvn_log1i = torch.stack([dmvnorm(X1, mu1i[i], cov1i, True) for i in range(k)], dim=1).to(device).detach()
        mvn_log2i = torch.stack([dmvnorm(X2, mu2i[i], cov2i, True) for i in range(k)], dim=1).to(device).detach()
        mvn_log3i = torch.stack([dmvnorm(X3, mu3i[i], cov3i, True) for i in range(k)], dim=1).to(device).detach()
        mvn_log4i = torch.stack([dmvnorm(X4, mu4i[i], cov4i, True) for i in range(k)], dim=1).to(device).detach()

        mvn_logi = mvn_log1i + mvn_log2i + mvn_log3i + mvn_log4i
        mvni = mvn_logi + prob_logi
        omega_logi = torch.log_softmax(mvni, dim=1)
        omegai = torch.exp(omega_logi)


        LL_mem = sixQ_function(par, prob_logi, omega_logi, data, k, n1, n2, n3, n4, times1, times2, times3, times4)

        nk = omegai.sum(dim=0)
        wk = nk.clamp_min(1e-12)
        pi = wk.pow(1.0 / max(2, 1e-8))
        pi = pi / pi.sum()

        alpha = 1 / (2 * k)
        prob = pi + alpha
        prob = prob / prob.sum()

        prob_logi = torch.log(prob)




        for _ in range(25):
            optimizer.zero_grad()
            Q = sixQ_function(par, prob_logi, omega_logi, data, k, n1, n2, n3, n4, times1, times2, times3, times4)
            Q.backward()
            torch.nn.utils.clip_grad_norm_(par, max_norm=1.0)
            optimizer.step()

        par_hat = par
        par = par_hat
        LL_next = sixQ_function(par, prob_logi, omega_logi, data, k, n1, n2, n3, n4,times1, times2, times3, times4)
        epsilon = (LL_next - LL_mem)/LL_next
        LL_mem = LL_next
        scheduler.step(float(LL_next.detach()))
        iter += 1

        final_assignments = torch.argmax(omegai, dim=1).cpu().numpy()

    with torch.no_grad():
        cov = get_sixSAD1(par[:8], n1, n2, n3, n4)
        pm = par[8:]
        means = torch.cat([
            linear_equation(t, torch.column_stack((pm[j*k:(j+1)*k], pm[(j+4)*k:(j+5)*k])))
            for j, t in enumerate((times1, times2, times3, times4))
        ], dim=1)
        joint = torch.stack([dmvnorm(data, means[j], cov) for j in range(k)], dim=1) + prob_logi
        omega_logi = torch.log_softmax(joint, dim=1)
        observed_nll = -torch.logsumexp(joint, dim=1).sum().item()
    prob_log = prob_logi.detach().cpu().numpy()
    omega_logi_cpu = omega_logi.cpu()
    max_values, max_indices = torch.max(omega_logi_cpu, dim=1)
    max_omega_logi = max_indices.detach().numpy()
    par = par.detach().cpu().numpy()
    LL_next = LL_next.detach().cpu().numpy()
    BIC = 2 * (LL_next) + np.log(d) * (len(par) + k - 1)
    return {
        'cluster_number': k,
        'log-likelihood': LL_next,
        'BIC': 2 * observed_nll + np.log(d) * (len(par) + k - 1),
        'legacy_Q_BIC': float(BIC),
        'negative_log_likelihood': observed_nll,
        'iterations': iter,
        'converged': bool(abs(float(epsilon.detach())) <= 0.0001),
        'posterior': torch.exp(omega_logi).cpu().numpy(),
        'par': par,
        'prob_log': prob_log,
        'max_omega_logi': max_omega_logi
    }

def main():
    global RUN_SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--iter-max", type=int, default=10000)
    args = parser.parse_args()
    if args.iter_max < 1:
        parser.error("--iter-max must be positive")
    root = Path(__file__).resolve().parents[1]
    input_dir = root / "results" / "01_read_data"
    output = root / "results" / "02_clustering"
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_dir / "data_all.csv", index_col=0)
    idx = pd.read_csv(input_dir / "index_all.csv", index_col=0)
    assert idx.shape == (1, df.shape[1]) and idx.columns.equals(df.columns)
    assert df.index.is_unique and df.columns.is_unique
    assert np.isfinite(df.to_numpy()).all() and np.isfinite(idx.to_numpy()).all()
    meta = pd.read_csv(root / "data" / "demo_metadata.csv").set_index("Sample")
    assert meta.index.is_unique
    groups = ["ColonT_NonMet", "ColonT_Liver", "ColonT_Lung", "ColonT_Other"]
    actual = meta.loc[df.columns, "Group"].tolist()
    sizes = [actual.count(g) for g in groups]
    assert all(n >= 2 for n in sizes)
    assert actual == [g for g, n in zip(groups, sizes) for _ in range(n)]
    offset = 0
    for n in sizes:
        assert np.all(np.diff(idx.to_numpy()[0, offset:offset+n]) >= 0)
        offset += n
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = torch.tensor(df.to_numpy(), dtype=torch.float32, device=device)
    time = torch.tensor(idx.to_numpy(), dtype=torch.float32, device=device)
    manifest = {"base_seed": args.seed, "device": str(device), "torch": torch.__version__,
                "groups": dict(zip(groups, sizes)), "data_shape": list(df.shape),
                "index_shape": list(idx.shape), "iter_max": args.iter_max,
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (input_dir / "data_all.csv", input_dir / "index_all.csv")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for k in range(2, 11):
        for run in range(1, 11):
            RUN_SEED = args.seed + k * 100 + run
            random.seed(RUN_SEED)
            np.random.seed(RUN_SEED)
            torch.manual_seed(RUN_SEED)
            torch.cuda.manual_seed_all(RUN_SEED)
            folder = output / f"k_{k:02d}" / f"run_{run:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Starting k={k}, run={run}, seed={RUN_SEED}", flush=True)
            result = sixfun_clu(data, k, time, *sizes, iter_max=args.iter_max)
            posterior = result.pop("posterior")
            labels = result["max_omega_logi"]
            assert posterior.shape == (len(df), k) and np.isfinite(posterior).all()
            assert np.allclose(posterior.sum(axis=1), 1, atol=1e-5)
            assert np.isfinite(result["par"]).all() and np.isfinite(result["BIC"])
            result.update(seed=RUN_SEED, run=run, protein_ids=df.index.tolist(),
                          cluster_counts=np.bincount(labels, minlength=k).tolist())
            serial = {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in result.items()}
            path = folder / "result.json"
            path.write_text(json.dumps(serial, indent=2, allow_nan=False), encoding="utf-8")
            pd.DataFrame({"Protein": df.index, "Cluster": labels + 1}).to_csv(folder / "labels.csv", index=False)
            pd.DataFrame(posterior, index=df.index, columns=[f"Cluster_{j+1}" for j in range(k)]).to_csv(folder / "posterior.csv", index_label="Protein")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            assert loaded == serial
            back = pd.read_csv(folder / "labels.csv")
            assert back.Protein.tolist() == df.index.tolist() and np.array_equal(back.Cluster, labels+1)
            pback = pd.read_csv(folder / "posterior.csv", index_col=0)
            assert pback.index.tolist() == df.index.tolist() and np.allclose(pback, posterior, atol=1e-7)
            rows.append({key: serial[key] for key in ("cluster_number", "run", "seed", "BIC", "negative_log_likelihood", "iterations", "converged")})
            pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
            print(f"Finished k={k}, run={run}, iterations={serial['iterations']}, BIC={serial['BIC']:.3f}", flush=True)
    summary = pd.DataFrame(rows)
    assert len(summary) == 90 and summary.groupby("cluster_number").size().eq(10).all()
    summary.loc[summary.groupby("cluster_number").BIC.idxmin()].to_csv(output / "best_per_k.csv", index=False)
    print("Verified 90 runs, protein order, finite results, and JSON/CSV round trips.", flush=True)


if __name__ == "__main__":
    main()
