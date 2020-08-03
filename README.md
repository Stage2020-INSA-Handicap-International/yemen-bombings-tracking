# yemen-bombings-tracking
Tracing a bombing map in Yemen

# Sommaire
## Architecture d'augmentation et de detection des bombardements grâce à Sentinel
### Sentinel
Sentinel est un groupe de satelittes gratuitement accessible de Agence spatiale européenne.
Les satelittes les plus precis sont les Sentinel-2A/B pour la qualité d'image et le Sentinel 5p pour les parametres atmospheriques (O3, CH4, NO2 etc...)

Le module fetcher est utilisé pour obtenir les données du satellite Sentinel-2.

--api-file : utilisé pour initialiser la connexion avec l'API sentinelsat avec un fichier json contenant "user" et "pass". Par défaut, utilise le document appelé SentinelAPIUser.json.
--district-file : json contenant une liste de tous les districts (dans notre cas tous les districts du Yémen) et leur polygone correspondant.
--district-name : nécessaire pour récupérer les informations d'un district particulier.
--start-date : format ddmmYYYY
--end-date : format ddmmYYYY
--level : utilisé pour indiquer le type de produit que nous recherchons. Par défaut, le niveau est 1C (type de produit S2MSI1C) mais le niveau peut être 2A (S2MSI2A)
--path: chemin vers le dossier où nous souhaitons sauvegarder les images telechargées.
La doc sentinelsat peut etre trouvé sur https://sentinelsat.readthedocs.io/en/stable/.

### Augmentation d'image
### Detection des bombardements
### Architecture
## Detection et comparaison d'images 
## Axes d'ameliorations et de recherches
