# Prédiction Profit Franchise

Un projet d'implémentation complète de régression linéaire pour prédire les profits de franchises de restaurants basé sur la population des villes.

## 📋 Vue d'ensemble du projet

Ce projet implémente un modèle de régression linéaire simple pour prédire les profits mensuels moyens d'une franchise de restaurant en fonction de la population de la ville. C'est une démonstration pratique des concepts fondamentaux du machine learning.

## 🎯 Objectif

Construire un modèle de régression linéaire capable de :
- Analyser la relation entre la population d'une ville et les profits d'un restaurant
- Prédire les profits potentiels pour de nouvelles villes candidates
- Identifier les opportunités d'expansion commerciale optimales

## 📊 Ensemble de données

- **Variables d'entrée (x)** : Population des villes (en unités de 10,000)
- **Variables de sortie (y)** : Profits mensuels (en unités de $10,000)
- **Nombre d'exemples** : 97 villes
- **Format** : Données NumPy (tableaux 1D)

## 🔧 Implémentation

### Fonction de coût
```
J(w,b) = (1/2m) Σ(f_w,b(x^(i)) - y^(i))²
```

### Gradient descent
```
∂J/∂b = (1/m) Σ(f_w,b(x^(i)) - y^(i))
∂J/∂w = (1/m) Σ(f_w,b(x^(i)) - y^(i))x^(i)
```

### Modèle linéaire
```
f_w,b(x) = wx + b
```

## 📁 Structure du projet

```
Prédiction-Profit-Franchise/
├── Prédiction Profit Franchise.ipynb    # Notebook principal
├── utils.py                              # Fonctions utilitaires
├── public_tests.py                       # Tests unitaires
├── README.md                             # Documentation
├── data/                                 # Données du projet
│   └── restaurant_data.txt
└── requirements.txt                      # Dépendances
```

## 🚀 Démarrage rapide

### Prérequis

```bash
Python 3.7+
NumPy
Matplotlib
```

### Installation

1. Cloner le repository
```bash
git clone https://github.com/votre-nom/Prediction-Profit-Franchise.git
cd Prediction-Profit-Franchise
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Ouvrir le notebook Jupyter
```bash
jupyter notebook "Prédiction Profit Franchise.ipynb"
```

## 💻 Utilisation

### Entraîner le modèle

```python
# Initialiser les paramètres
initial_w = 0.
initial_b = 0.

# Hyperparamètres
iterations = 1500
alpha = 0.01  # Taux d'apprentissage

# Entraîner le modèle
w, b, _, _ = gradient_descent(
    x_train, y_train, 
    initial_w, initial_b, 
    compute_cost, 
    compute_gradient, 
    alpha, 
    iterations
)
```

### Faire des prédictions

```python
# Prédire le profit pour une ville de 35,000 habitants
population = 3.5  # 35,000 / 10,000
profit = w * population + b
print(f'Profit prédit: ${profit*10000:.2f}')
```

## 📈 Résultats

Après 1500 itérations avec un taux d'apprentissage de 0.01 :

| Métrique | Valeur |
|----------|--------|
| Paramètre w | 1.166 |
| Paramètre b | -3.630 |
| Coût final | 4.49 |
| Population 35,000 | $4,519.77 |
| Population 70,000 | $45,342.45 |

## 🧪 Tests

Exécuter les tests unitaires :

```python
from public_tests import *

# Test de la fonction de coût
compute_cost_test(compute_cost)

# Test du gradient
compute_gradient_test(compute_gradient)
```

### Résultats attendus

```
✓ Cost at initial w: 75.203
✓ Gradient at initial w, b (zeros): -65.329, -5.839
✓ All tests passed!
```

## 📚 Concepts clés

- **Régression linéaire** : Modèle statistique pour prédire une variable continue
- **Fonction de coût** : Mesure la performance du modèle (erreur quadratique moyenne)
- **Gradient descent** : Algorithme d'optimisation itérative
- **Paramètres** : Poids (w) et biais (b) du modèle
- **Taux d'apprentissage** : Contrôle la taille des pas d'optimisation

## 📊 Visualisations

Le projet inclut :
- Graphique en nuage de points des données d'entraînement
- Ligne de régression linéaire ajustée
- Convergence de la fonction de coût

## 🔬 Améliorations possibles

- Ajouter une validation croisée
- Implémenter une régularisation (L1/L2)
- Étendre à la régression multiple (plusieurs variables)
- Ajouter de la normalisation des données
- Comparer avec d'autres modèles (polynomial, Ridge, Lasso)

## 📝 Notes techniques

- Les données sont stockées sous forme de tableaux NumPy 1D
- L'implémentation utilise uniquement NumPy (pas de scikit-learn)
- Le modèle utilise la descente de gradient batch (tous les exemples)
- La convergence est garantie pour ce problème simple et convexe

## 👥 Apprentissage pédagogique

Ce projet démontre :
- Implémentation manuelle d'algorithmes de ML
- Compréhension du calcul différentiel appliqué
- Visualisation et interprétation des résultats
- Bonnes pratiques de programmation Python

## 📄 Licence

Ce projet est sous licence MIT.

## 🤝 Contributions

Les contributions sont bienvenues ! N'hésitez pas à :
- Soumettre des pull requests
- Signaler des bugs
- Proposer des améliorations

## 📧 Contact

Pour toute question ou suggestion, veuillez ouvrir une issue sur GitHub.

---

**Créé avec ❤️ pour l'apprentissage du machine learning**
