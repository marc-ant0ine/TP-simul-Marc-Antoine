import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Paramètres
N = 500  # taille de l'échantillon à simuler
n = 100  # taille de chaque somme pour le TCL


mu = 0.5
sigma = np.sqrt(1/12)  # écart-type de U([0,1])

# Génération des échantillons N(0,1) par TCL
Z_unif = np.zeros(N)
for i in range(N):
   
    X = np.random.uniform(0, 1, n)
    
    S_n = np.sum(X)
  
    Z_unif[i] = (S_n - n*mu) / (sigma*np.sqrt(n))

print(f"Statistiques de l'échantillon généré (loi uniforme) :")
print(f"Moyenne : {np.mean(Z_unif):.4f} (théorique : 0)")
print(f"Variance : {np.var(Z_unif):.4f} (théorique : 1)")
print(f"Écart-type : {np.std(Z_unif):.4f}")

# Visualisation de la qualité de la simulation
plt.figure(figsize=(14, 10))

# Histogramme de Z avec densité normale superposée
plt.subplot(2, 2, 1)
plt.hist(Z_unif, bins=30, density=True, alpha=0.7, 
         edgecolor='black', color='skyblue', label='Échantillon TCL')
x = np.linspace(-4, 4, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, 
         label='N(0,1) théorique')
plt.xlabel('z')
plt.ylabel('Densité')
plt.title(f'Histogramme - Loi Uniforme\n(n={n}, N={N})')
plt.legend()
plt.grid(True, alpha=0.3)

# Fonction de répartition empirique vs théorique
plt.subplot(2, 2, 2)
x_sorted = np.sort(Z_unif)
y_empirical = np.arange(1, N+1) / N
plt.plot(x_sorted, y_empirical, 'b-', linewidth=2, 
         label='Fonction de répartition empirique')
plt.plot(x, stats.norm.cdf(x, 0, 1), 'r-', linewidth=2, 
         label='Fonction de répartition N(0,1)')
plt.xlabel('z')
plt.ylabel('Probabilité cumulée')
plt.title('Fonction de répartition')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Avec loi exponentielle λ=1
lambda_exp = 1
mu_exp = 1/lambda_exp  # μ = 1
sigma_exp = 1/lambda_exp  # σ = 1

# Génération des échantillons N(0,1) par TCL
Z_exp = np.zeros(N)
for i in range(N):
    # Générer n variables exponentielles i.i.d.
    X_exp = np.random.exponential(scale=1/lambda_exp, size=n)
    # Calculer S_n = somme des X_i
    S_n_exp = np.sum(X_exp)
    # Calculer Z_n selon le TCL
    Z_exp[i] = (S_n_exp - n*mu_exp) / (sigma_exp*np.sqrt(n))

print(f"\n" + "="*60)
print(f"Statistiques de l'échantillon généré (loi exponentielle) :")
print(f"Moyenne : {np.mean(Z_exp):.4f} (théorique : 0)")
print(f"Variance : {np.var(Z_exp):.4f} (théorique : 1)")
print(f"Écart-type : {np.std(Z_exp):.4f}")

# Visualisation pour la loi exponentielle
plt.figure(figsize=(14, 10))

# Histogramme
plt.subplot(2, 2, 1)
plt.hist(Z_exp, bins=30, density=True, alpha=0.7, 
         edgecolor='black', color='lightgreen', label='Échantillon TCL')
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, 
         label='N(0,1) théorique')
plt.xlabel('z')
plt.ylabel('Densité')
plt.title(f'Histogramme - Loi Exponentielle λ=1\n(n={n}, N={N})')
plt.legend()
plt.grid(True, alpha=0.3)

# Fonction de répartition
plt.subplot(2, 2, 2)
x_sorted_exp = np.sort(Z_exp)
y_empirical_exp = np.arange(1, N+1) / N
plt.plot(x_sorted_exp, y_empirical_exp, 'g-', linewidth=2, 
         label='Fonction de répartition empirique')
plt.plot(x, stats.norm.cdf(x, 0, 1), 'r-', linewidth=2, 
         label='Fonction de répartition N(0,1)')
plt.xlabel('z')
plt.ylabel('Probabilité cumulée')
plt.title('Fonction de répartition')
plt.legend()
plt.grid(True, alpha=0.3)
