import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
from scipy.special import erf, erfinv


def box_muller_normal(n):
    """Génère n variables N(0,1) par Box-Muller."""
    u = np.random.rand(n)
    v = np.random.rand(n)
   
    z = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    return z

def tcl_uniform_normal(n, n_sum=100):
    """Génère n variables N(0,1) par TCL avec loi uniforme."""
    z = np.zeros(n)
    mu = 0.5 
    sigma = np.sqrt(1/12)  
    
    for i in range(n):
       
        uniforms = np.random.rand(n_sum)
        somme = np.sum(uniforms)
      
        z[i] = (somme - n_sum * mu) / (sigma * np.sqrt(n_sum))
    return z

def tcl_exponential_normal(n, n_sum=100, lambd=1):
    """Génère n variables N(0,1) par TCL avec loi exponentielle."""
    z = np.zeros(n)
    mu = 1/lambd 
    sigma = 1/lambd  
    
    for i in range(n):
      
        uniforms = np.random.rand(n_sum)
        exponentials = -np.log(uniforms) / lambd
        somme = np.sum(exponentials)
       
        z[i] = (somme - n_sum * mu) / (sigma * np.sqrt(n_sum))
    return z

def inversion_normal(n):
    """Génère n variables N(0,1) par inversion (approximation de Moro)."""
    u = np.random.rand(n)
  
    z = stats.norm.ppf(u)
    return z



N = 500
methods = {
    'Box-Muller': box_muller_normal,
    'TCL Uniforme': lambda n: tcl_uniform_normal(n, n_sum=100),
    'TCL Exponentielle': lambda n: tcl_exponential_normal(n, n_sum=100),
    'Inversion': inversion_normal
}


results = {}
execution_times = {}

print("=" * 60)
print("COMPARAISON DES MÉTHODES POUR N(0,1)")
print("=" * 60)

for method_name, method_func in methods.items():
    print(f"\n{method_name}:")
    print("-" * 30)
    

    start_time = time.time()
    sample = method_func(N)
    end_time = time.time()
    

    mean_val = np.mean(sample)
    var_val = np.var(sample, ddof=1)
    
    ks_stat, ks_pval = stats.kstest(sample, 'norm', args=(0, 1))
    

    results[method_name] = {
        'sample': sample,
        'mean': mean_val,
        'variance': var_val,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval
    }
    
    execution_times[method_name] = (end_time - start_time) * 1000  # en ms
    
   
    print(f"Moyenne empirique: {mean_val:.4f}")
    print(f"Variance empirique: {var_val:.4f}")
    print(f"Test KS - D: {ks_stat:.4f}, p-value: {ks_pval:.4f}")
    print(f"Temps d'exécution: {execution_times[method_name]:.2f} ms")


mu = 3.5
sigma = 2.2
N_transformed = 500

print("\n" + "=" * 60)
print(f"SIMULATION DE N({mu}, {sigma**2:.2f}) PAR BOX-MULLER")
print("=" * 60)


start_time = time.time()
z_box = box_muller_normal(N_transformed) 
x_transformed = mu + sigma * z_box 
end_time = time.time()


mean_trans = np.mean(x_transformed)
var_trans = np.var(x_transformed, ddof=1)


ks_stat_trans, ks_pval_trans = stats.kstest(x_transformed, 'norm', args=(mu, sigma))


rel_error_mean = abs(mean_trans - mu) / mu * 100
rel_error_var = abs(var_trans - sigma**2) / sigma**2 * 100

print(f"\nParamètres théoriques: μ = {mu}, σ² = {sigma**2:.4f}")
print(f"Moyenne empirique: {mean_trans:.4f} (écart: {rel_error_mean:.2f}%)")
print(f"Variance empirique: {var_trans:.4f} (écart: {rel_error_var:.2f}%)")
print(f"Test KS - D: {ks_stat_trans:.4f}, p-value: {ks_pval_trans:.4f}")
print(f"Temps d'exécution: {(end_time - start_time)*1000:.2f} ms")




fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Comparaison des méthodes pour générer N(0,1)', fontsize=14)

for idx, (method_name, result) in enumerate(results.items()):
    ax = axes1[idx // 2, idx % 2]
    sample = result['sample']
    
  
    ax.hist(sample, bins=30, density=True, alpha=0.6, color='skyblue', 
            edgecolor='black', label='Empirique')
    

    x_range = np.linspace(-4, 4, 1000)
    ax.plot(x_range, stats.norm.pdf(x_range), 'r-', linewidth=2, 
            label='Théorique N(0,1)')
    

    stats_text = (f'Moyenne: {result["mean"]:.3f}\n'
                  f'Variance: {result["variance"]:.3f}\n'
                  f'KS D: {result["ks_stat"]:.3f}\n'
                  f'Temps: {execution_times[method_name]:.1f} ms')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Densité')
    ax.set_title(f'{method_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()


fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('QQ-plots des différentes méthodes', fontsize=14)

for idx, (method_name, result) in enumerate(results.items()):
    ax = axes2[idx // 2, idx % 2]
    sample = result['sample']
    
 
    stats.probplot(sample, dist="norm", plot=ax)
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)
    
    ax.set_title(f'QQ-plot - {method_name}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()


fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(f'Simulation de N({mu}, {sigma**2:.2f}) par Box-Muller', fontsize=14)


axes3[0].hist(x_transformed, bins=30, density=True, alpha=0.6, 
              color='lightgreen', edgecolor='black', label='Empirique')


x_range_trans = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
axes3[0].plot(x_range_trans, stats.norm.pdf(x_range_trans, mu, sigma), 
              'r-', linewidth=2, label='Théorique')

axes3[0].set_xlabel('Valeur')
axes3[0].set_ylabel('Densité')
axes3[0].set_title(f'Histogramme de N({mu}, {sigma**2:.2f})')
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)


stats.probplot(x_transformed, dist="norm", plot=axes3[1])
axes3[1].get_lines()[0].set_marker('o')
axes3[1].get_lines()[0].set_markersize(4)
axes3[1].get_lines()[0].set_alpha(0.6)
axes3[1].get_lines()[1].set_color('red')
axes3[1].get_lines()[1].set_linewidth(2)
axes3[1].set_title(f'QQ-plot de N({mu}, {sigma**2:.2f})')
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()


fig4, ax4 = plt.subplots(figsize=(10, 4))
fig4.suptitle('Temps d\'exécution des différentes méthodes', fontsize=14)


methods_names = list(execution_times.keys())
times = list(execution_times.values())

bars = ax4.bar(methods_names, times, color=['skyblue', 'lightcoral', 
                                            'lightgreen', 'gold'])
ax4.set_ylabel('Temps (ms)')
ax4.set_xlabel('Méthode')


for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{time_val:.1f} ms', ha='center', va='bottom')

ax4.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

print("\n" + "=" * 60)
print("TABLEAU RÉCAPITULATIF DES MÉTHODES")
print("=" * 60)
print("\nMéthode            | Moyenne  | Variance | KS D     | p-value  | Temps (ms)")
print("-" * 75)

for method_name in methods.keys():
    r = results[method_name]
    print(f"{method_name:18} | {r['mean']:7.4f} | {r['variance']:8.4f} | "
          f"{r['ks_stat']:7.4f} | {r['ks_pval']:7.4f} | {execution_times[method_name]:9.2f}")

print("\n" + "=" * 60)
print("RÉSUMÉ DE LA TRANSFORMATION N(0,1) -> N(mu, sigma²)")
print("=" * 60)
print(f"\nParamètres : μ = {mu}, σ = {sigma}, σ² = {sigma**2:.4f}")
print(f"Moyenne empirique : {mean_trans:.4f} (écart relatif : {rel_error_mean:.2f}%)")
print(f"Variance empirique : {var_trans:.4f} (écart relatif : {rel_error_var:.2f}%)")
print(f"Test KS : D = {ks_stat_trans:.4f}, p-value = {ks_pval_trans:.4f}")



plt.show()
