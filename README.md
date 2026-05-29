# Prédiction des contraintes dans un milieu granulaire par réseau de neurones de graphes

## Description

Ce projet a pour objectif de prédire les forces de contact et les chaînes de forces dans un milieu granulaire bidimensionnel à partir de simulations réalisées par la Méthode des Éléments Discrets (DEM).

Les données générées par les simulations sont représentées sous forme de graphes et utilisées pour entraîner un réseau de neurones de graphes (GNN) basé sur l'architecture GraphSAGE afin de prédire les forces de contact entre particules.

## Structure du dépôt

- `code de generation_donnees_DEM.py` : génération des données à partir des simulations DEM.
- `Code d'entrainement_GNN.py` : entraînement du modèle GNN.
- `code de validation_GNN.py` : validation du modèle et analyse des performances.
- `minidem.py` : bibliothèque utilisée pour les simulations DEM.

## Méthodes utilisées

- Méthode des Éléments Discrets (DEM)
- Représentation du milieu granulaire sous forme de graphe
- Réseau de neurones de graphes (GNN)
- Architecture GraphSAGE
- Optimisation Adam
- Fonction de perte MSE

## Résultats

Le modèle permet de prédire les principales structures des chaînes de forces et de retrouver les zones dominantes de transmission des contraintes dans le milieu granulaire.

## Auteur

Amadou OUATTARA

Master Energie

Université de Lorraine

Année universitaire 2025-2026
