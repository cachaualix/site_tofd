# TOFD – site web

Site de présentation du projet STG2 : extraction automatique des extrémités de défaut sur un B-scan TOFD.

## Voir le site

**https://cachaualix.github.io/site_tofd/**

Le lien fonctionne une fois GitHub Pages activé (Settings → Pages → branche `main`, dossier `/ (root)`).

## Contenu du site

- Une démo interactive sur un B-scan simulé (tips à 380 et 490 échantillons)
- Le principe de la TOFD, avec un schéma
- La méthode en 5 étapes : regroupement, débruitage, enveloppe, séparation des pics, vérification
- Les limites connues de l'outil

## Structure du dépôt

| Élément | Rôle |
|---|---|
| `index.html` | Le site (page unique) |
| `backend/` | Code Python du pipeline (`tofd_pipeline.py`) et de l'interface (`app.py`) |

## Modifier le site en local

Ouvre `index.html` dans un navigateur. Aucune installation n'est nécessaire.

## Auteur

[cachaualix](https://github.com/cachaualix)
