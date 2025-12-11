import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy.special import comb, factorial
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InverseMethodSimulation:
    def __init__(self, N=10000, seed=42):
        """
        Initialise la simulation avec la méthode inverse
        
        Parameters:
        N: taille de l'échantillon
        seed: seed pour la reproductibilité
        """
        np.random.seed(seed)
        self.N = N
        
    # Fonction quantile pour loi Binomiale
    def binomial_quantile(self, u, n, p):
        """
        Fonction quantile pour la loi Binomiale(n, p)
        
        Parameters:
        u: valeur uniforme dans [0,1]
        n: nombre d'essais
        p: probabilité de succès
        """
        cum_prob = 0
        for k in range(n + 1):
            cum_prob += self.binomial_pmf(k, n, p)
            if cum_prob >= u:
                return k
        return n
    
    def binomial_pmf(self, k, n, p):
        """PMF de la loi binomiale"""
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    # Fonction quantile pour loi de Poisson
    def poisson_quantile(self, u, lambda_):
        """
        Fonction quantile pour la loi de Poisson(λ)
        """
        cum_prob = 0
        k = 0
        while cum_prob < u:
            cum_prob += self.poisson_pmf(k, lambda_)
            if cum_prob >= u:
                return k
            k += 1
            if k > 1000:  # Sécurité
                break
        return k
    
    def poisson_pmf(self, k, lambda_):
        """PMF de la loi de Poisson"""
        return np.exp(-lambda_) * (lambda_ ** k) / factorial(k)
    
    # Fonction quantile pour loi Exponentielle
    def exponential_quantile(self, u, theta):
        """
        Fonction quantile pour la loi Exponentielle(θ)
        F(x) = 1 - exp(-θx)
        F^(-1)(u) = -log(1-u)/θ
        """
        return -np.log(1 - u) / theta
    
    # Fonction quantile pour f(x) = 2x sur [0,1]
    def custom_quantile(self, u):
        """
        Fonction quantile pour f(x) = 2x sur [0,1]
        F(x) = x²
        F^(-1)(u) = sqrt(u)
        """
        return np.sqrt(u)
    
    def generate_sample(self, quantile_func, *args):
        """
        Génère un échantillon de taille N
        
        Parameters:
        quantile_func: fonction quantile
        *args: paramètres pour la fonction quantile
        """
        u = np.random.uniform(0, 1, self.N)
        sample = np.array([quantile_func(u_i, *args) for u_i in u])
        return sample
    
    def compute_statistics(self, sample):
        """
        Calcule les statistiques descriptives
        """
        return {
            'moyenne': np.mean(sample),
            'variance': np.var(sample),
            'ecart_type': np.std(sample),
            'min': np.min(sample),
            'max': np.max(sample),
            'median': np.median(sample),
            'skewness': stats.skew(sample),
            'kurtosis': stats.kurtosis(sample)
        }
    
    def ks_test(self, sample, cdf_func):
        """
        Test de Kolmogorov-Smirnov
        """
        sorted_sample = np.sort(sample)
        n = len(sorted_sample)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = np.array([cdf_func(x) for x in sorted_sample])
        
        D_plus = np.max(empirical_cdf - theoretical_cdf)
        D_minus = np.max(theoretical_cdf - empirical_cdf)
        D = max(D_plus, D_minus)
        
        return D, D_plus, D_minus
    
    def cdf_binomial(self, x, n, p):
        """CDF de la loi binomiale"""
        return sum(self.binomial_pmf(k, n, p) for k in range(int(np.floor(x)) + 1))
    
    def cdf_poisson(self, x, lambda_):
        """CDF de la loi de Poisson"""
        return sum(self.poisson_pmf(k, lambda_) for k in range(int(np.floor(x)) + 1))
    
    def cdf_exponential(self, x, theta):
        """CDF de la loi exponentielle"""
        return 1 - np.exp(-theta * x) if x >= 0 else 0
    
    def cdf_custom(self, x):
        """CDF pour f(x) = 2x"""
        return x ** 2 if 0 <= x <= 1 else (1 if x > 1 else 0)
    
    def run_simulation(self):
        """
        Exécute la simulation complète pour toutes les distributions
        """
        results = {}
        
        # 1. Binomiale (n=20, p=0.3)
        print("Simulation de la loi Binomiale(20, 0.3)...")
        binomial_sample = self.generate_sample(self.binomial_quantile, 20, 0.3)
        binomial_stats = self.compute_statistics(binomial_sample)
        binomial_D, binomial_D_plus, binomial_D_minus = self.ks_test(
            binomial_sample, lambda x: self.cdf_binomial(x, 20, 0.3)
        )
        
        results['binomial'] = {
            'sample': binomial_sample,
            'stats': binomial_stats,
            'theory': {'mean': 20 * 0.3, 'variance': 20 * 0.3 * 0.7},
            'ks': {'D': binomial_D, 'D_plus': binomial_D_plus, 'D_minus': binomial_D_minus}
        }
        
        # 2. Poisson (λ=5)
        print("Simulation de la loi de Poisson(5)...")
        poisson_sample = self.generate_sample(self.poisson_quantile, 5)
        poisson_stats = self.compute_statistics(poisson_sample)
        poisson_D, poisson_D_plus, poisson_D_minus = self.ks_test(
            poisson_sample, lambda x: self.cdf_poisson(x, 5)
        )
        
        results['poisson'] = {
            'sample': poisson_sample,
            'stats': poisson_stats,
            'theory': {'mean': 5, 'variance': 5},
            'ks': {'D': poisson_D, 'D_plus': poisson_D_plus, 'D_minus': poisson_D_minus}
        }
        
        # 3. Exponentielle (θ=2)
        print("Simulation de la loi Exponentielle(2)...")
        exp_sample = self.generate_sample(self.exponential_quantile, 2)
        exp_stats = self.compute_statistics(exp_sample)
        exp_D, exp_D_plus, exp_D_minus = self.ks_test(
            exp_sample, lambda x: self.cdf_exponential(x, 2)
        )
        
        results['exponential'] = {
            'sample': exp_sample,
            'stats': exp_stats,
            'theory': {'mean': 1/2, 'variance': 1/(2**2)},
            'ks': {'D': exp_D, 'D_plus': exp_D_plus, 'D_minus': exp_D_minus}
        }
        
        # 4. Loi custom f(x) = 2x
        print("Simulation de la loi f(x)=2x sur [0,1]...")
        custom_sample = self.generate_sample(self.custom_quantile)
        custom_stats = self.compute_statistics(custom_sample)
        custom_D, custom_D_plus, custom_D_minus = self.ks_test(
            custom_sample, self.cdf_custom
        )
        
        results['custom'] = {
            'sample': custom_sample,
            'stats': custom_stats,
            'theory': {'mean': 2/3, 'variance': 1/18},
            'ks': {'D': custom_D, 'D_plus': custom_D_plus, 'D_minus': custom_D_minus}
        }
        
        return results
    
    def plot_results(self, results):
        """
        Crée les visualisations pour les résultats
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Histogrammes
        distributions = ['binomial', 'poisson', 'exponential', 'custom']
        titles = ['Loi Binomiale(20, 0.3)', 'Loi de Poisson(5)', 
                  'Loi Exponentielle(2)', 'Loi f(x)=2x sur [0,1]']
        
        for i, (dist, title) in enumerate(zip(distributions, titles), 1):
            sample = results[dist]['sample']
            stats = results[dist]['stats']
            theory = results[dist]['theory']
            
            ax = plt.subplot(4, 4, i)
            if dist in ['binomial', 'poisson']:
                # Distributions discrètes
                unique, counts = np.unique(sample, return_counts=True)
                ax.bar(unique, counts/len(sample), alpha=0.7, 
                       label='Empirique', width=0.8 if dist == 'binomial' else 0.6)
                
                # Distribution théorique
                if dist == 'binomial':
                    x_theo = np.arange(0, 21)
                    y_theo = [self.binomial_pmf(k, 20, 0.3) for k in x_theo]
                else:  # poisson
                    x_theo = np.arange(0, 15)
                    y_theo = [self.poisson_pmf(k, 5) for k in x_theo]
                
                ax.plot(x_theo, y_theo, 'ro-', markersize=4, label='Théorique', linewidth=2)
            else:
                # Distributions continues
                ax.hist(sample, bins=50, density=True, alpha=0.7, 
                       label='Empirique', edgecolor='black')
                
                # Distribution théorique
                if dist == 'exponential':
                    x_theo = np.linspace(0, np.max(sample), 1000)
                    y_theo = 2 * np.exp(-2 * x_theo)
                    ax.plot(x_theo, y_theo, 'r-', linewidth=2, label='Théorique')
                else:  # custom
                    x_theo = np.linspace(0, 1, 1000)
                    y_theo = 2 * x_theo
                    ax.plot(x_theo, y_theo, 'r-', linewidth=2, label='Théorique')
            
            # Ajout des statistiques dans le titre
            ax.set_title(f'{title}\n'
                        f'Moyenne: {stats["moyenne"]:.3f} (théorique: {theory["mean"]:.3f})\n'
                        f'Variance: {stats["variance"]:.3f} (théorique: {theory["variance"]:.3f})',
                        fontsize=10)
            ax.set_xlabel('Valeur')
            ax.set_ylabel('Densité/Fréquence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. QQ-plots
        for i, (dist, title) in enumerate(zip(distributions, titles), 5):
            ax = plt.subplot(4, 4, i)
            sample = results[dist]['sample']
            
            if dist == 'binomial':
                theo_dist = stats.binom(n=20, p=0.3)
            elif dist == 'poisson':
                theo_dist = stats.poisson(mu=5)
            elif dist == 'exponential':
                theo_dist = stats.expon(scale=0.5)  # scale = 1/θ
            else:  # custom
                theo_dist = None
            
            if theo_dist is not None:
                stats.probplot(sample, dist=theo_dist, plot=ax)
            else:
                # QQ-plot pour la distribution custom
                sorted_sample = np.sort(sample)
                theo_quantiles = np.sqrt(np.random.uniform(0, 1, len(sample)))
                theo_quantiles = np.sort(theo_quantiles)
                
                ax.scatter(theo_quantiles, sorted_sample, alpha=0.5)
                ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
                ax.set_xlabel('Quantiles théoriques')
                ax.set_ylabel('Quantiles empiriques')
            
            ax.set_title(f'QQ-plot - {title}')
            ax.grid(True, alpha=0.3)
        
        # 3. Résultats du test KS
        ax_ks = plt.subplot(4, 4, 13)
        ks_data = []
        for dist, title in zip(distributions, titles):
            ks_data.append({
                'Distribution': title,
                'Statistique D': results[dist]['ks']['D'],
                'D+': results[dist]['ks']['D_plus'],
                'D-': results[dist]['ks']['D_minus']
            })
        
        ks_df = pd.DataFrame(ks_data)
        ax_ks.axis('tight')
        ax_ks.axis('off')
        table = ax_ks.table(cellText=ks_df.round(4).values,
                           colLabels=ks_df.columns,
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax_ks.set_title('Test de Kolmogorov-Smirnov', fontsize=12, pad=20)
        
        # 4. Histogramme des valeurs uniformes initiales
        ax_uniform = plt.subplot(4, 4, 14)
        u_values = np.random.uniform(0, 1, self.N)
        ax_uniform.hist(u_values, bins=50, density=True, alpha=0.7, 
                       edgecolor='black', color='skyblue')
        ax_uniform.axhline(y=1, color='r', linestyle='--', linewidth=2, 
                          label='Uniforme théorique')
        ax_uniform.set_xlabel('Valeur')
        ax_uniform.set_ylabel('Densité')
        ax_uniform.set_title(f'Distribution des {self.N} valeurs uniformes')
        ax_uniform.legend()
        ax_uniform.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_table(self, results):
        """
        Affiche un tableau récapitulatif des statistiques
        """
        print("\n" + "="*100)
        print("RÉCAPITULATIF DES SIMULATIONS - MÉTHODE INVERSÉE")
        print("="*100)
        
        summary_data = []
        for dist_name, dist_results in results.items():
            stats = dist_results['stats']
            theory = dist_results['theory']
            
            summary_data.append({
                'Distribution': dist_name.upper(),
                'N échantillon': self.N,
                'Moyenne empirique': f"{stats['moyenne']:.4f}",
                'Moyenne théorique': f"{theory['mean']:.4f}",
                'Différence moyenne': f"{abs(stats['moyenne'] - theory['mean']):.4f}",
                'Variance empirique': f"{stats['variance']:.4f}",
                'Variance théorique': f"{theory['variance']:.4f}",
                'Différence variance': f"{abs(stats['variance'] - theory['variance']):.4f}",
                'KS D-stat': f"{dist_results['ks']['D']:.4f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        print("\n" + "="*100)
        print("INTERPRÉTATION DU TEST DE KOLMOGOROV-SMIRNOV:")
        print("="*100)
        print("H₀: L'échantillon suit la distribution théorique")
        print("Plus D est petit, plus l'échantillon suit bien la distribution théorique.")
        print("La valeur critique pour α=0.05 et N=10000 est environ 0.0135")
        print("Si D > 0.0135, on rejette H₀ au seuil de 5%")
        print("="*100)

# Exécution de la simulation
if __name__ == "__main__":
    # Création de la simulation
    simulator = InverseMethodSimulation(N=10000, seed=42)
    
    # Exécution des simulations
    print("Démarrage de la simulation par méthode inverse...")
    print(f"Taille d'échantillon: {simulator.N}")
    print("-" * 50)
    
    results = simulator.run_simulation()
    
    # Affichage des résultats
    simulator.print_summary_table(results)
    
    # Visualisations
    print("\nGénération des visualisations...")
    simulator.plot_results(results)
